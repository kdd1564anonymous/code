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
from utils import count_parameters, get_logger, close_logger, search_for_trails


def search(args):
    if args.benchmark == '101':
        from search.nas_101_utils import spec2data, NASBench, Architect, OPS_FULL, N_NODES, MAX_EDGES
        from nasbench import api
    else:
        import nas_201_api as nb
        from search.nas_201_utils import train_and_eval, CifarBench, Architect, arch2data
        OP_SPACE = cell.SearchSpaceNames['nas-bench-201']
        N_NODES = 4

    def initialize_pool(bench, size):
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
            bench.eval_arch(arch_str)
            arch_pool.append(arch)
            # time_cost += cost

        return arch_pool, seen_arch

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

    if args.benchmark == '101':
        nas_bench = api.NASBench(args.nas_bench_path)
        cifar_bench = NASBench(nas_bench, average_all=args.average_all)
        predictor = Predictor(t_edge=1,
                              t_node=len(OPS_FULL),
                              n_node=N_NODES,
                              h_dim=64,
                              n_out=1).to('cuda')

        def enum_arch_data():
            for h in cifar_bench._all_hashes:
                yield h, spec2data(cifar_bench.hash2spec(h))

    elif args.benchmark == '201':
        nas_bench = nb.NASBench201API(args.nas_bench_path)
        predictor = Predictor(len(OP_SPACE), 1, N_NODES, 64, 1).to('cuda')
        cifar_bench = CifarBench(nas_bench)

        def enum_arch_data():
            duplicated = set()
            for idx in range(len(nas_bench)):
                archstr = nas_bench[idx]
                struct = gt.Structure.str2structure(archstr)
                unique_str = struct.to_unique_str(True)
                if unique_str not in duplicated:
                    duplicated.add(unique_str)
                    yield archstr, arch2data(archstr)

    optim_p = torch.optim.Adam(predictor.parameters(),
                               args.p_lr,
                               weight_decay=args.weight_decay)

    logger.info("params size = %fM" % (count_parameters(predictor) / 1e6))
    logger.info("\n")

    logger.info("initialize arch pool")
    arch_pool, seen_arch = initialize_pool(cifar_bench, args.pool_size)

    history = [cifar_bench.arch_str(a.struct) for a in arch_pool]

    # logging initial samples
    best_arch_seen = cifar_bench.choose_best(seen_arch)
    logger.info("init pool: %d, seen arch: %d" %
                (len(arch_pool), len(seen_arch)))
    logger.info("simulated time cost: %f" % cifar_bench.total_cost)
    logger.info("best initial arch:")
    cifar_bench.log_arch(best_arch_seen, 0, 'acc_best', logger, writer)

    logger.info('start training predictor')

    train_loader = gd.DataListLoader(cifar_bench.history_data(),
                                     args.train_batch_size,
                                     shuffle=True)

    for epoch in tqdm(range(args.epochs)):
        loss = predictor.fit(train_loader, optim_p, 0, None, args.regression,
                             args.grad_clip, 0)
        writer.add_scalar('loss_r', loss, epoch)

    logger.info('preparing valid data')
    all_arch, all_data = list(
        zip(*tqdm(filter(lambda v: v[0] not in seen_arch, enum_arch_data()))))
    pred_results = []
    pred_loader = gd.DataLoader(all_data, batch_size=args.step_batch_size)
    with torch.no_grad():
        for batch in tqdm(pred_loader, total=len(pred_loader)):
            batch = batch.to('cuda')
            pred_results.append(predictor(batch).cpu().numpy())
    pred_results = np.concatenate(pred_results, axis=0).flatten()
    arg_rank = np.argsort(pred_results)

    while cifar_bench.total_cost < args.time_budget:
        cur_index = arg_rank[0]
        logger.info("current time cost: %f" % cifar_bench.total_cost)
        logger.info("arch to eval: %s" % all_arch[cur_index])
        cifar_bench.eval_arch(all_arch[cur_index])
        history.append(all_arch[cur_index])
        arg_rank = arg_rank[1:]

    best_arch_seen = cifar_bench.choose_best(history)

    with open(os.path.join(logdir, 'selections'), 'w') as f:
        for a in history:
            f.write(a + ',' + str(cifar_bench.lookup(a)))
            f.write('\n')

    return best_arch_seen, cifar_bench.valid_acc(
        best_arch_seen), cifar_bench.test_acc(best_arch_seen)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark',
                        type=str,
                        choices=['101', '201'],
                        required=True)
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument('--pool_size', type=int, default=60)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--step_batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--p_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--nas_bench_path', type=str, required=True)
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--log_dir', type=str, default='logs/searches-npnas')
    parser.add_argument('--time_budget', type=float, default=5e5)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--workers', type=int, default=1)

    args = parser.parse_args()

    search_for_trails(search, args)
