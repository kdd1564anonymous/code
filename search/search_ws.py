import torch
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
import argparse

from torch import tensor
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tqdm import tqdm
from operator import itemgetter

from predictor import Predictor
from search.nas_201_utils import arch2data, arch2acc, train_and_eval, Architect, CifarBench
from architect.search_model import TinyNetwork
from utils import AverageMeter, count_parameters, get_logger, obtain_accuracy, time_string, search_for_trails, grouper, visualize_all_rank, close_logger
from .dataset import get_datasets, split_dataset
from .ws_utils import *


# def search_find_best(xloader, train_loader, network, bn_iters, predictor,
#                      optimizer, n_total, n_init, n_limit, p_batch_size,
#                      p_epochs, logger):
def search_find_best(xloader, train_loader, network, predictor,
                     optimizer, logger, args):
    bn_iter = 0 if not args.track_running_stat else args.bn_train_iters
    arch_pool = [Architect() for _ in range(args.init_pool_size)]
    valid_accs = {}
    best_arch, best_acc = None, 100.
    # network.eval()
    loader_iter = iter(xloader)
    while True:
        logger.info('arch pool: %d' % len(arch_pool))
        # evaluate unseen architectures
        with torch.no_grad():
            for arch in arch_pool:
                # arch_str = arch.struct.to_unique_str(True)
                arch_str = arch.struct.tostr()
                if arch_str in valid_accs:
                    continue
                network.set_genotype(arch.struct)

                if bn_iter > 0:
                    train_bn(train_loader, network, bn_iter)
                network.eval()

                arch_top1 = AverageMeter()
                try:
                    inputs, targets = next(loader_iter)
                except:
                    loader_iter = iter(xloader)
                    inputs, targets = next(loader_iter)

                inputs, targets = inputs.cuda(), targets.cuda()
                _, logits = network(inputs)
                val_top1 = obtain_accuracy(logits, targets.data)[0]
                arch_top1.update(val_top1.item(), targets.size(0))

                arch_err = 100. - arch_top1.avg
                valid_accs[arch_str] = arch_err
                if arch_err < best_acc:
                    best_arch = arch.struct.tostr()
                    best_acc = arch_err

        logger.info("best arch err ever: %2.2f" % best_acc)
        logger.info(best_arch)
        if len(arch_pool) >= args.max_samples:
            break

        # train predictor
        p_train_data = [
            arch2data(
                a.struct.tostr(),
                #   valid_accs[a.struct.to_unique_str(True)])
                valid_accs[a.struct.tostr()]) for a in arch_pool
        ]
        p_train_queue = gd.DataListLoader(p_train_data,
                                          args.p_batch_size,
                                          shuffle=True)
        for epoch in range(args.p_epochs):
            # train_predictor(predictor, p_train_data, optimizer, p_batch_size)
            predictor.fit(p_train_queue, optimizer, 0, None)

        # grad search
        checker = lambda arch: arch.struct.tostr() not in valid_accs
        new_trace = predictor.grad_step_on_archs(arch_pool, args.step_batch_size, args.step_size, checker)
        new_trace = sorted(new_trace, key=itemgetter(1))[:args.new_pop_limit]

        # new_trace = sorted(new_trace, key=itemgetter(1))[:args.max_samples]
        arch_pool = (arch_pool + list(map(itemgetter(0), new_trace)))[:args.max_samples]

    return best_arch, best_acc


# def main(args, nas_bench, logger):
def main(args):
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
        logger.info('{:20} : {:}'.format(name, value))

    nas_bench = nb.NASBench201API(args.nas_bench_path)
    op_space = cell.SearchSpaceNames[args.search_space_name]
    predictor = Predictor(len(op_space), 1, args.max_nodes, 64, 1).to('cuda')
    optim_p = torch.optim.Adam(predictor.parameters(),
                               args.p_lr,
                               weight_decay=args.weight_decay)

    logger.info("predictor params size = %fM" %
                (count_parameters(predictor) / 1e6))
    logger.info("\n")

    train_data, _, _, classnum = get_datasets(args.dataset, args.data_path,
                                              -1)  # disable cutout for search
    train_data, valid_data = split_dataset(args.dataset, train_data)
    supernet = TinyNetwork(args.channel, args.num_cells, args.max_nodes,
                                 classnum, op_space, False,
                                 args.track_running_stat).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optim = torch.optim.SGD(supernet.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, args.epochs, args.eta_min)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               args.batch_size,
                                               True,
                                               num_workers=args.load_workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               args.batch_size_valid,
                                               True,
                                               drop_last=True,
                                               num_workers=args.load_workers,
                                               pin_memory=True)

    start_time, search_time, epoch_time = time.time(), AverageMeter(
    ), AverageMeter()
    if not args.load_checkpoint:
        # first, train supernet by uniform sampling
        for epoch in range(args.start_epoch, args.epochs):
            loss, top1, top5 = train_supernet(train_loader,
                                              supernet, criterion, optim,
                                              str(epoch), args.print_freq,
                                              logger)

            scheduler.step()
            writer.add_scalar('train/loss', loss, epoch)
            writer.add_scalar('train/top1', top1, epoch)
            writer.add_scalar('train/top5', top5, epoch)
            loss, top1, top5 = valid_func(valid_loader, supernet, criterion)
            logger.info("Valid: loss=%.2f, top1=%2.2f, top5=%2.2f" %
                        (loss, top1, top5))
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        # save trained supernet weights
        torch.save(supernet.state_dict(),
                   os.path.join(args.log_dir, 'supernet.pth'))
    else:
        logger.info('load supernet from %s' % args.load_checkpoint)
        load_path = os.path.join(args.load_checkpoint, 'supernet.pth')

        supernet.load_state_dict(torch.load(load_path, map_location='cuda'))
        logger.info('supernet loaded')

    search_time.update(epoch_time.sum)
    logger.info('Pre-searching costs {:.1f} s'.format(search_time.sum))
    start_time = time.time()
    # perform search
    best_arch, best_valid_acc = search_find_best(
        valid_loader, train_loader, supernet, predictor, optim_p,
        logger, args)
    search_time.update(time.time() - start_time)
    arch_idx = nas_bench.query_index_by_arch(best_arch)
    logger.info('found best arch with valid error %f:' % best_valid_acc)
    nas_bench.show(arch_idx)
    logger.info('time cost: %fs' % search_time.sum)

    vacc = 100 - nas_bench.get_more_info(arch_idx, 'cifar10-valid', None,
                                         is_random=False, hp='200')['valid-accuracy']
    tacc = 100 - nas_bench.get_more_info(arch_idx, 'cifar10', None,
                                         is_random=False, hp='200')['test-accuracy']

    close_logger(logger)

    return best_arch, vacc, tacc, search_time.sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                        help='Choose between Cifar10/100 and ImageNet-16.')
    # channels and number-of-cells
    parser.add_argument('--search_space_name',
                        type=str,
                        default='nas-bench-201',
                        help='The search space name.')
    parser.add_argument('--max_nodes',
                        type=int,
                        default=4,
                        help='The maximum number of nodes.')
    parser.add_argument('--channel',
                        type=int,
                        default=16,
                        help='The number of channels.')
    parser.add_argument('--num_cells',
                        type=int,
                        default=5,
                        help='The number of cells in one stage.')
    parser.add_argument('--track_running_stat',
                        type=int,
                        default=1,
                        choices=[0, 1],
                        help='The number of cells in one stage.')
    parser.add_argument('--bn_train_iters', type=int, default=50)

    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--eta_min', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_valid', type=int, default=1024)
    parser.add_argument('--nesterov', type=bool, default=1)

    parser.add_argument('--p_lr', type=float, default=1e-3)
    parser.add_argument('--p_batch_size', type=int, default=32)
    parser.add_argument('--p_epochs', type=int, default=50)
    parser.add_argument('--p_weight_decay', type=float, default=5e-4)
    parser.add_argument('--new_pop_limit', type=int, default=8)
    parser.add_argument('--init_pool_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--step_size', type=float, default=1.)
    parser.add_argument('--step_batch_size', type=int, default=128)
    # parser.add_argument('--eval_batches', type=int, default=10)

    parser.add_argument('--load_workers',
                        type=int,
                        default=0,
                        help='number of data loading workers')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs/searches-ws/%s' % time_string(),
                        help='Folder to save checkpoints and log.')
    parser.add_argument('--nas_bench_path',
                        default=None,
                        type=str,
                        help='The path to load NAS-Bench-201.')
    parser.add_argument('--print_freq',
                        type=int,
                        default=200,
                        help='print frequency (default: 200)')
    parser.add_argument('--seed', type=int, default=114514, help='manual seed')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()

    # search_for_trails(main, args)
    log_dir = args.log_dir
    if args.tag:
        log_dir += args.tag
    os.makedirs(log_dir, exist_ok=True)

    if args.repeat > 1:
        random.seed(args.seed)
        seeds = random.sample(range(args.repeat * 10), args.repeat)
    else:
        seeds = [args.seed]

    results = []
    time_costs = []
    ckpt_root = args.load_checkpoint
    for i in range(args.repeat):
        args.seed = seeds[i]
        args.log_dir = os.path.join(log_dir, str(i))
        if ckpt_root:
            args.load_checkpoint = os.path.join(ckpt_root, str(i))
        arch, vacc, tacc, search_time = main(args)
        with open(os.path.join(log_dir, 'archs.txt'), mode='a') as f:
            f.write(arch + '\n')
        results.append((arch, vacc, tacc))
        time_costs.append(search_time)

    logger = get_logger(os.path.join(log_dir, 'log_overall'))
    for arch, vacc, tacc in results:
        logger.info(arch)
        logger.info("valid=%f, test=%f"%(vacc, tacc))
    if args.repeat > 1:
        archs, val_accs, test_accs = zip(*results)
        logger.info('searched architectures:')
        for a in archs:
            logger.info(a)

        logger.info('result:')
        logger.info('val acc: %2.2f ~ %.2f' %
                    (np.mean(val_accs), np.std(val_accs)))
        logger.info('test acc: %2.2f ~ %.2f' %
                    (np.mean(test_accs), np.std(test_accs)))
        logger.info('avg time cost: %fs' % np.mean(time_costs))
