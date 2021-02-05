import torch
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
import torch.nn as nn
import torch_geometric
import torch_geometric.data as gd
import numpy as np
import random
import architect.genotypes as gt
import architect.cell_operations as cell
import time
import os
import logging
import torch.multiprocessing as mp
import copy
import functools

from torch import tensor
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tqdm import tqdm
from operator import itemgetter

from predictor import Predictor
from search.nas_101_utils import spec2data, NASBench, Architect, OPS_FULL, N_NODES, MAX_EDGES
from utils import grouper, AverageMeter, count_parameters, get_logger, plot_rank, visualize_all_rank, close_logger, search_for_trails
from nasbench import api


def initialize_pool(bench: NASBench, size):
    arch_pool = []
    seen_arch = set()
    # time_cost = 0.

    while len(seen_arch) < size:
        arch = Architect().randomize_()
        # unique_str = arch.struct.to_unique_str(True)
        struct = arch.struct
        arch_str = bench.arch_str(struct)
        while arch_str is None or arch_str in seen_arch:
            arch.randomize_()
            struct = arch.struct
            arch_str = bench.arch_str(struct)
        seen_arch.add(arch_str)
        bench.eval_arch(struct)
        arch_pool.append(arch)
        # time_cost += cost

    return arch_pool, seen_arch


def merge_arch_pool(bench: NASBench, origin_arch, new_arch, cost_limit,
                    pool_size):
    new_pool = []
    selected = []
    pid = []
    for arch, p in new_arch:
        struct = arch.struct
        acc = bench.eval_arch(struct)
        new_pool.append(arch)
        selected.append((arch, acc))
        pid.append(p)
        if bench.total_cost > cost_limit:
            break

    new_pool = new_pool + origin_arch

    return new_pool, selected, pid


def search(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logdir = args.log_dir
    writer = SummaryWriter(log_dir=logdir)

    logger = get_logger(os.path.join(logdir, 'log'))
    logger.info('Arguments : -------------------------------')
    for name, value in args._get_kwargs():
        logger.info('{:16} : {:}'.format(name, value))

    predictor = Predictor(t_edge=1,
                          t_node=len(OPS_FULL),
                          n_node=N_NODES,
                          h_dim=64,
                          n_out=1,
                          n_layers=4).to('cuda')

    optim_p = torch.optim.Adam(predictor.parameters(),
                               args.p_lr,
                               weight_decay=args.weight_decay)

    logger.info("params size = %fM" % (count_parameters(predictor) / 1e6))
    logger.info("\n")

    nas_bench = api.NASBench(args.nas_bench_path)
    cifar_bench = NASBench(nas_bench,
                           average_all=args.average_all,
                           use_test=args.use_test)

    logger.info("initialize arch pool")
    arch_pool, seen_arch = initialize_pool(cifar_bench, args.pool_size)

    # logging initial samples
    best_arch_seen = cifar_bench.choose_best(seen_arch)
    logger.info("init pool: %d, seen arch: %d" %
                (len(arch_pool), len(seen_arch)))
    logger.info("simulated time cost: %f" % cifar_bench.total_cost)
    logger.info("best initial arch:")
    cifar_bench.log_arch(best_arch_seen, 0, 'acc_best', logger, writer)

    trace_history = []
    select_history = [[(arch, cifar_bench.lookup(arch.struct))
                       for arch in arch_pool]]
    pid = [-1 for _ in arch_pool]
    if args.regression:
        _y = [cifar_bench.lookup(arch.struct) for arch in arch_pool]
        _mean = np.mean(_y)
        _std = np.std(_y)

        def reg_norm(y):
            return (y - _mean) / _std
    else:
        reg_norm = None

    global_train_step = 0
    for step in range(1, args.steps + 1):
        # train predictor
        logger.info('step on predictor')
        # use valid acc for predictor training
        train_loader = gd.DataListLoader(cifar_bench.history_data(),
                                         args.train_batch_size,
                                         shuffle=True)
        for _ in tqdm(range(args.epochs)):
            loss = predictor.fit(train_loader, optim_p, args.unlabeled_size,
                                 args.grad_clip)
            writer.add_scalar('loss_r', loss, global_train_step)
            global_train_step += 1

        # step on arch
        logger.info('step on arch')

        def checker(arch):
            arch_str = cifar_bench.arch_str(arch.struct)
            return arch_str is not None and arch_str not in seen_arch

        step_size = args.step_size
        new_trace = []
        while len(new_trace) == 0 and step_size < args.step_size * (2**3):
            new_trace = predictor.grad_step_on_archs(arch_pool,
                                                     args.step_batch_size,
                                                     step_size,
                                                     checker,
                                                     log_conn=False)
            step_size *= 2

        # select new population according to predicted acc
        new_trace = sorted(new_trace, key=itemgetter(1))[:args.new_pop_limit]
        new_arch = [(t[0], t[-1]) for t in new_trace]
        old_arch = sorted(arch_pool,
                          key=lambda a: cifar_bench.lookup(a.struct))

        logger.info("produced %d new archs" % len(new_trace))
        trace_history.append(new_trace)

        arch_pool, selected, new_pid = merge_arch_pool(cifar_bench, old_arch,
                                                       new_arch,
                                                       args.time_budget,
                                                       args.pool_size)

        select_history.append(selected)
        seen_arch.update(cifar_bench.arch_str(a.struct) for a, _ in selected)
        pid += new_pid

        logger.info("step %d, new archs: %d, seen arch %d" %
                    (step, len(arch_pool), len(seen_arch)))
        logger.info("simulated time cost: %f" % cifar_bench.total_cost)

        if len(selected) > 0:
            best_arch_select, best_acc_select = min(selected,
                                                    key=itemgetter(1))
            best_arch_select = cifar_bench.arch_str(best_arch_select.struct)
            logger.info("best arch of current step:")
            cifar_bench.log_arch(best_arch_select, step, 'acc_step', logger,
                                 writer)

            if best_acc_select < cifar_bench.lookup(best_arch_seen):
                best_arch_seen = best_arch_select

        logger.info("best arch ever:")
        cifar_bench.log_arch(best_arch_seen, step, 'acc_best', logger, writer)
        if cifar_bench.total_cost >= args.time_budget:
            break

    with open(os.path.join(logdir, 'selections'), 'w') as f:
        for step in select_history:
            for a in step:
                f.write(cifar_bench.arch_str(a[0].struct) + ',' + str(a[1]))
                f.write('\n')

    with open(os.path.join(logdir, 'pid'), 'w') as f:
        for p in pid:
            f.write(str(p))
            f.write('\n')

    predictor.save(os.path.join(logdir, 'predictor.pth'))
    close_logger(logger)

    return best_arch_seen, cifar_bench.valid_acc(
        best_arch_seen), cifar_bench.test_acc(best_arch_seen)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument('--pool_size', type=int, default=64)
    parser.add_argument('--new_pop_limit', type=int, default=5)
    parser.add_argument('--step_batch_size', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--step_size', type=float, default=1.)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--p_lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--nas_bench_path', type=str, required=True)
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--log_dir', type=str, default='logs/searches-101')
    parser.add_argument('--time_budget', type=float, default=5e5)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--workers', type=int, default=1)

    args = parser.parse_args()

    search_for_trails(search, args)


if __name__ == '__main__':
    main()
