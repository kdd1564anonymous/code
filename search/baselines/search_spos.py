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
import copy

from torch import tensor
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tqdm import tqdm
from operator import itemgetter

from search.nas_201_utils import arch2data, arch2acc, train_and_eval, Architect, CifarBench
from architect.search_model import TinyNetwork
from utils import AverageMeter, count_parameters, get_logger, obtain_accuracy, time_string, search_for_trails, grouper, visualize_all_rank, close_logger
from search.dataset import get_datasets, split_dataset

OP_SPACE = cell.SearchSpaceNames['nas-bench-201']
N_NODES = 4

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))


# from https://github.com/megvii-model/SinglePathOneShot/blob/master/src/Search/search.py
class EvolutionSearcher(object):
    def __init__(self, supernet, max_nodes, op_names, valid_func, logger, args):
        self.args = args
        self.logger = logger

        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num

        self.log_dir = args.log_dir
        self.supernet = supernet

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

        self.max_nodes, self.op_names = max_nodes, op_names
        self.valid_func = valid_func

    def is_legal(self, cand):
        # assert isinstance(cand, tuple) and len(cand) == self.nr_layer
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        self.logger.info(cand)
        info['err'] = self.valid_func(self.supernet, cand, self.args)

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        self.logger.info('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random(self, num):
        self.logger.info('random select ........')

        def random_architecture():
            genotypes = []
            for i in range(1, self.max_nodes):
                xlist = []
                for j in range(i):
                    op_name = random.choice(self.op_names)
                    xlist.append((op_name, j))
                genotypes.append(tuple(xlist))
            return gt.Structure(genotypes)

        cand_iter = self.stack_random_cand(random_architecture)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            self.logger.info('random {}/{}'.format(len(self.candidates), num))
        self.logger.info('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        self.logger.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            child_arch = copy.deepcopy(cand)
            node_id = random.randint(0, len(child_arch.nodes) - 1)
            node_info = list(child_arch.nodes[node_id])
            snode_id = random.randint(0, len(node_info) - 1)
            xop = random.choice(self.op_names)
            while xop == node_info[snode_id][0]:
                xop = random.choice(self.op_names)
            node_info[snode_id] = (xop, node_info[snode_id][1])
            child_arch.nodes[node_id] = tuple(node_info)
            return child_arch

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            self.logger.info('mutation {}/{}'.format(len(res), mutation_num))

        self.logger.info('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1, _ = choice(self.keep_top_k[k]).tolist(None)
            p2, _ = choice(self.keep_top_k[k]).tolist(None)
            child = []
            for node1, node2 in zip(p1, p2):
                xlist = []
                for (op1, pre), (op2, _) in zip(node1, node2):
                    xlist.append((choice([op1, op2]), pre))
                child.append(xlist)
            return gt.Structure(child)
            # return tuple(choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            self.logger.info('crossover {}/{}'.format(len(res), crossover_num))

        self.logger.info('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        self.logger.info(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'
            .format(
                self.population_num, self.select_num, self.mutation_num,
                self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num,
                self.max_epochs))

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            self.logger.info('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(self.candidates,
                              k=self.select_num,
                              key=lambda x: self.vis_dict[x]['err'])
            self.update_top_k(self.candidates,
                              k=50,
                              key=lambda x: self.vis_dict[x]['err'])

            self.logger.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                self.logger.info('No.{} {} Top-1 err = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['err']))
                ops = [i for i in cand]
                self.logger.info(ops)

            mutation = self.get_mutation(self.select_num, self.mutation_num,
                                         self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

        best_arch = self.keep_top_k[self.select_num][0]
        return best_arch, self.vis_dict[best_arch]['err']


def train_supernet(xloader, network, criterion, w_optimizer, epoch_str,
                   print_freq, logger):
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(
    ), AverageMeter()
    network.train()
    end = time.time()
    for step, (X, Y) in enumerate(xloader):
        # measure data loading time
        X, Y = X.cuda(non_blocking=True), Y.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        # update the weights
        network.random_genotype(True)
        w_optimizer.zero_grad()
        _, logits = network(X)
        base_loss = criterion(logits, Y)
        base_loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 5)
        w_optimizer.step()
        # record
        base_prec1, base_prec5 = obtain_accuracy(logits.data,
                                                 Y.data,
                                                 topk=(1, 5))
        base_losses.update(base_loss.item(), X.size(0))
        base_top1.update(base_prec1.item(), X.size(0))
        base_top5.update(base_prec5.item(), X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(xloader):
            Sstr = '*SEARCH* ' + ' [{:}][{:03d}/{:03d}]'.format(
                epoch_str, step, len(xloader))
            Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(
                batch_time=batch_time, data_time=data_time)
            Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(
                loss=base_losses, top1=base_top1, top5=base_top5)
            logger.info(Sstr + ' ' + Tstr + ' ' + Wstr)
    return base_losses.avg, base_top1.avg, base_top5.avg


def valid_func(xloader, network, criterion):
    data_time, batch_time = AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(
    ), AverageMeter()
    end = time.time()
    with torch.no_grad():
        for step, (X, Y) in enumerate(xloader):
            X = X.cuda(non_blocking=True)
            Y = Y.cuda(non_blocking=True)
            # measure data loading time
            data_time.update(time.time() - end)
            # prediction

            network.random_genotype(True)
            _, logits = network(X)
            arch_loss = criterion(logits, Y)
            # record
            arch_prec1, arch_prec5 = obtain_accuracy(logits.data,
                                                     Y.data,
                                                     topk=(1, 5))
            arch_losses.update(arch_loss.item(), Y.size(0))
            arch_top1.update(arch_prec1.item(), Y.size(0))
            arch_top5.update(arch_prec5.item(), Y.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return arch_losses.avg, arch_top1.avg, arch_top5.avg


@torch.no_grad()
def train_bn(loader, network, n_iter):
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.zero_()
            m.running_var.zero_()
    network.train()
    loader = iter(loader)
    for i in range(n_iter):
        inputs, _ = next(loader)
        inputs = inputs.cuda(non_blocking=True)
        network(inputs)


def search_find_best(xloader, train_loader, network, logger, args):
    op_space = cell.SearchSpaceNames[args.search_space_name]
    loader_iter = iter(xloader)

    bn_iter = 0 if not args.track_running_stat else args.bn_train_iters

    @torch.no_grad()
    def valid_fn(network, arch, args):
        network.arch_cache = arch
        train_bn(iter(train_loader), network, bn_iter)
        network.eval()
        try:
            inputs, targets = next(loader_iter)
        except:
            loader_iter = iter(xloader)
            inputs, targets = next(loader_iter)

        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(
            non_blocking=True)
        _, logits = network(inputs)
        val_top1 = obtain_accuracy(logits, targets.data)[0]
        return val_top1

    searcher = EvolutionSearcher(network, args.max_nodes, op_space, valid_fn,
                                 logger, args)
    return searcher.search()


nas_bench = None


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

    global nas_bench
    logger = get_logger(os.path.join(logdir, 'log'))
    logger.info('Arguments : -------------------------------')
    for name, value in args._get_kwargs():
        logger.info('{:20} : {:}'.format(name, value))

    op_space = cell.SearchSpaceNames[args.search_space_name]

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
        # try:
        #     d = int(os.path.basename(logdir))
        #     load_path = os.path.join(args.load_checkpoint, str(d), 'supernet.pth')
        # except:
        #     load_path = os.path.join(args.load_checkpoint, 'supernet.pth')
        load_path = os.path.join(args.load_checkpoint, 'supernet.pth')

        supernet.load_state_dict(torch.load(load_path, map_location='cuda'))
        logger.info('supernet loaded')

    search_time.update(epoch_time.sum)
    logger.info('Pre-searching costs {:.1f} s'.format(search_time.sum))
    start_time = time.time()
    # perform search
    best_arch, best_valid_acc = search_find_best(valid_loader, train_loader, supernet,
                                                 logger, args)
    search_time.update(time.time() - start_time)
    arch_idx = nas_bench.query_index_by_arch(best_arch)
    logger.info('found best arch with valid error %f:' % best_valid_acc)
    nas_bench.show(arch_idx)
    logger.info('time cost: %fs' % search_time.sum)

    vacc = 100 - nas_bench.get_more_info(arch_idx, 'cifar10-valid', None,
                                         is_random=False)['valid-accuracy']
    tacc = 100 - nas_bench.get_more_info(arch_idx, 'cifar10', None,
                                         is_random=False)['test-accuracy']


    close_logger(logger)

    return best_arch.tostr(), vacc, tacc, search_time.sum


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

    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=10)
    parser.add_argument('--m_prob', type=float, default=0.1)
    parser.add_argument('--crossover-num', type=int, default=5)
    parser.add_argument('--mutation-num', type=int, default=5)

    parser.add_argument('--load_workers',
                        type=int,
                        default=0,
                        help='number of data loading workers')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs/searches-spos/%s' % time_string(),
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
    nas_bench = nb.NASBench201API(args.nas_bench_path)
    log_dir = args.log_dir
    os.makedirs(args.log_dir, exist_ok=True)

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

    logger = get_logger(os.path.join(args.log_dir, 'log_overall'))
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
