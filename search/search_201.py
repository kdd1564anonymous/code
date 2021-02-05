import torch
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
import torch.multiprocessing as mp
import torch.nn as nn
import torch_geometric
import torch_geometric.data as gd
import numpy as np
import random
import nas_201_api as nb
import architect.genotypes as gt
import architect.cell_operations as cell
import time
import os
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
from search.nas_201_utils import train_and_eval, CifarBench, Architect, arch2data
from utils import grouper, AverageMeter, count_parameters, get_logger, plot_rank, visualize_all_rank, close_logger, search_for_trails

OP_SPACE = cell.SearchSpaceNames['nas-bench-201']
N_NODES = 4


def initialize_pool(bench: CifarBench, size):
    arch_pool = []
    seen_arch = set()
    # time_cost = 0.

    while len(seen_arch) < size:
        arch = Architect().randomize_()
        # unique_str = arch.struct.to_unique_str(True)
        arch_str = arch.struct.tostr()
        while arch_str in seen_arch:
            arch.randomize_()
            arch_str = arch.struct.tostr()
        seen_arch.add(arch_str)
        bench.eval_arch(arch_str)
        arch_pool.append(arch)
        # time_cost += cost

    return arch_pool, seen_arch


def merge_arch_pool(bench: CifarBench, origin_arch, new_arch, cost_limit,
                    pool_size):
    new_pool = []
    selected = []
    pid = []
    for arch, p in new_arch:
        arch_str = arch.struct.tostr()
        acc, cost = bench.eval_arch(arch_str)
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

    nas_bench = nb.NASBench201API(args.nas_bench_path)

    logger.info('Arguments : -------------------------------')
    for name, value in args._get_kwargs():
        logger.info('{:16} : {:}'.format(name, value))

    predictor = Predictor(len(OP_SPACE), 1, N_NODES, 64, 1, n_layers=3).to('cuda')
    optim_p = torch.optim.Adam(predictor.parameters(),
                               args.p_lr,
                               weight_decay=args.weight_decay)

    logger.info("params size = %fM" % (count_parameters(predictor) / 1e6))
    logger.info("\n")

    cifar_bench = CifarBench(nas_bench)

    def log_arch(arch, step, name):
        valid_12_acc = cifar_bench.lookup(arch)
        valid_final_acc = cifar_bench.valid_acc(arch)
        test_acc = cifar_bench.test_acc(arch)

        logger.info('\n' + nas_bench.query_by_arch(arch))

        writer.add_scalar('%s/valid12' % name, valid_12_acc, step)
        writer.add_scalar('%s/valid' % name, valid_final_acc, step)
        writer.add_scalar('%s/test' % name, test_acc, step)

    logger.info("initialize arch pool")
    arch_pool, seen_arch = initialize_pool(cifar_bench, args.pool_size)

    # logging initial samples
    best_arch_seen = cifar_bench.choose_best(seen_arch)
    logger.info("init pool: %d, seen arch: %d" %
                (len(arch_pool), len(seen_arch)))
    logger.info("simulated time cost: %f" % cifar_bench.total_cost)
    logger.info("best initial arch:")
    log_arch(best_arch_seen, 0, 'acc_best')

    trace_history = []
    # select_history = []
    select_history = [[(arch, cifar_bench.lookup(arch.struct.tostr())) for arch in arch_pool]]
    pid = [-1 for _ in arch_pool]
    global_train_step = 0
    for step in range(1, args.steps + 1):
        # train predictor
        logger.info('step on predictor')
        # use valid acc for predictor training
        arch_data = cifar_bench.history_data()
        train_loader = gd.DataListLoader(cifar_bench.history_data(),
                                         args.train_batch_size,
                                         shuffle=True)
        for _ in tqdm(range(args.epochs)):
            random.shuffle(arch_data)
            loss = predictor.fit(train_loader, optim_p, args.grad_clip, args.decoder_coe)
            writer.add_scalar('loss_r', loss, global_train_step)
            global_train_step += 1

        # step on arch
        def checker(arch):
            return arch.struct.tostr() not in seen_arch

        logger.info('step on arch')
        step_size = args.step_size
        new_trace = []
        while len(new_trace) == 0 and step_size < args.step_size * (2**3):
            new_trace = predictor.grad_step_on_archs(arch_pool,
                                                     args.step_batch_size,
                                                     step_size, checker)
            step_size *= 2

        # select new population according to predicted acc
        new_trace = sorted(new_trace, key=itemgetter(1))[:args.new_pop_limit]
        new_arch = [(t[0], t[-1]) for t in new_trace]
        old_arch = sorted(arch_pool,
                          key=lambda a: cifar_bench.lookup(a.struct.tostr()))

        logger.info("produced %d new archs" % len(new_trace))
        trace_history.append(new_trace)

        arch_pool, selected, new_pid = merge_arch_pool(cifar_bench, old_arch, new_arch,
                                              args.time_budget, args.pool_size)

        select_history.append(selected)
        seen_arch.update(a.struct.tostr() for a, _ in selected)
        pid += new_pid

        logger.info("step %d, new archs: %d, seen arch %d" %
                    (step, len(arch_pool), len(seen_arch)))
        logger.info("simulated time cost: %f" % cifar_bench.total_cost)

        if len(selected) > 0:
            best_arch_select, best_acc_select = min(selected,
                                                    key=itemgetter(1))
            best_arch_select = best_arch_select.struct.tostr()
            logger.info("best arch of current step:")
            log_arch(best_arch_select, step, 'acc_step')

            if best_acc_select < cifar_bench.lookup(best_arch_seen):
                best_arch_seen = best_arch_select

        logger.info("best arch ever:")
        log_arch(best_arch_seen, step, 'acc_best')
        if cifar_bench.total_cost >= args.time_budget:
            break

    with open(os.path.join(logdir, 'selections'), 'w') as f:
        for step in select_history:
            for a in step:
                f.write(a[0].struct.tostr() + ',' + str(a[1]))
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
    parser.add_argument('--pool_size', type=int, default=32)
    parser.add_argument('--new_pop_limit', type=int, default=5)
    parser.add_argument('--step_batch_size', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--step_size', type=float, default=1.)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--p_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument(
        '--nas_bench_path',
        type=str,
        default='/home/disk/nas-bench-201/NAS-Bench-201-v1_1-096897.pth')
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--log_dir', type=str, default='logs/searches')
    parser.add_argument('--time_budget', type=int, default=30000)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--tag', type=str, default=None)

    args = parser.parse_args()

    search_for_trails(search, args)


if __name__ == '__main__':
    main()
