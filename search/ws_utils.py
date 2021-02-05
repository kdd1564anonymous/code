import torch

import torch.nn as nn
import torch_geometric
import torch_geometric.data as gd
import numpy as np
import random
import architect.genotypes as gt
import architect.cell_operations as cell
import time
import os
import argparse

from torch import tensor
from tqdm import tqdm
from operator import itemgetter

from utils import AverageMeter, obtain_accuracy, time_string
from .dataset import get_datasets, split_dataset


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
    network.eval()
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
