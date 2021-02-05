import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch_geometric.data as gd
import numpy as np
import logging
import time
import torch.multiprocessing as mp
import copy
import functools
import os


from matplotlib import pyplot as plt

def grouper(n, iterable, drop_last=False):
    if drop_last:
        args = [iter(iterable)] * n
        return zip(*args)
    else:
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]


def plot_rank(pred, gt):
    pred, gt = np.array(pred), np.array(gt)
    rank_on_pred = np.argsort(np.argsort(pred))
    rank_on_gt = np.argsort(np.argsort(gt))
    fig, ax = plt.subplots()
    ax.scatter(rank_on_pred, rank_on_gt, s=1)
    ax.plot([0, len(gt)], [0, len(gt)], 'r--')
    return fig


@torch.no_grad()
def visualize_all_rank(model, valid_data, fn, batch_size):
    pred = []
    gt = []
    loader = gd.DataLoader(valid_data, batch_size, shuffle=False)

    model.eval()
    for X in loader:
        gt.append(X.y)
        X = X.to('cuda')
        pred.append(model(X).detach().cpu())
    pred = torch.cat(pred, dim=0).view(-1).numpy()
    gt = torch.cat(gt, dim=0).view(-1).numpy()
    fig = plot_rank(pred, gt)
    fig.savefig(fn)



class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.val = val

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('nas')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

def obtain_accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

def time_string():
    return time.strftime("%Y%m%d-%H%M%S")


def search_for_trails(search, args):
    logdir = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"))
    if args.tag:
        logdir += '-' + args.tag
    os.makedirs(logdir, exist_ok=True)

    if args.repeat > 1:
        random.seed(args.seed)
        seeds = random.sample(range(args.repeat * 10), args.repeat)
    else:
        seeds = [args.seed]

    results = []
    if args.repeat > 1 and args.workers > 1:
        args_arr = [copy.deepcopy(args) for _ in range(args.repeat)]
        for i in range(args.repeat):
            args_arr[i].seed = seeds[i]
            args_arr[i].log_dir = os.path.join(logdir, str(i))

        # search_func = functools.partial(search, nas_bench)
        search_func = search
        with mp.Pool(args.workers) as p:
            results = p.map(search_func, args_arr)
    else:
        for i in range(args.repeat):
            args.seed = seeds[i]
            args.log_dir = os.path.join(logdir, str(i))
            arch, vacc, tacc = search(args)
            results.append((arch, vacc, tacc))

    if args.repeat > 1:
        logger = get_logger(os.path.join(logdir, 'log_overall'))
        archs, val_accs, test_accs = zip(*results)

        logger.info('searched architectures:')
        for a in archs:
            logger.info(a)

        logger.info('result:')
        logger.info('val acc: %.4f ~ %.4f' %
                    (np.mean(val_accs), np.std(val_accs)))
        logger.info('test acc: %.4f ~ %.4f' %
                    (np.mean(test_accs), np.std(test_accs)))
    return results
