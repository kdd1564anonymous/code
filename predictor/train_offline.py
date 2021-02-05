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

from torch import tensor
from predictor import Predictor, MLP
from .alphax import AlphaxMLP

from scipy import stats
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter

from search.nas_201_utils import OP_IDX
from utils import grouper, random_interpolate, plot_rank, AverageMeter

from matplotlib import pyplot as plt


@torch.no_grad()
def valid(valid_data, model, batch_size):
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
    return pred, gt


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logdir = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"))
    logdir += '-%s-%s'%(args.model, args.dataset)
    if args.tag:
        logdir += '-' + args.tag
    writer = SummaryWriter(log_dir=logdir)

    if args.dataset == '201':
        train_data = torch.load('data/nas201-train.pth')
        valid_data = torch.load('data/nas201-valid.pth')
    else:
        train_data = torch.load('data/nas101-train.pth')
        valid_data = torch.load('data/nas101-valid.pth')
        valid_data = random.sample(valid_data, 1000)

    random.shuffle(train_data)
    train_data_labeled = train_data[:args.samples]

    if args.model == 'gnn' and args.dataset == '101':
        model = Predictor(t_edge=1,
                        t_node=5,
                        n_node=7,
                        h_dim=128,
                        n_out=1).to('cuda')
    elif args.model == 'gnn' and args.dataset == '201':
        model = Predictor(len(OP_IDX), 1, 4, 64, 1).to('cuda')
    elif args.model == 'mlp' and args.dataset == '201':
        model = MLP(len(OP_IDX), 6, 64, 1).to('cuda')
    else:
        model = AlphaxMLP().to('cuda')

    if args.checkpoint:
        print("load checkpoint from %s"%args.checkpoint)
        model.load(args.checkpoint)
        pred, gt = valid(valid_data, model, args.val_batch_size)
        rho, pval = stats.spearmanr(pred, gt)
        print('rho=%f, pval=%f' % (rho, pval))
        fig_rank = plot_rank(pred, gt)
        plt.savefig(os.path.join(logdir, 'rank.png'))
        plt.close(fig_rank)

        return

    optim = torch.optim.Adam(model.parameters(),
                             args.lr,
                             weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epoch)

    labeled_loader = gd.DataListLoader(train_data_labeled,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=4)
    for step in range(args.epoch):
        model.train()
        loss_r, loss_d = AverageMeter(), AverageMeter()
        loss = model.fit(labeled_loader, optim, args.grad_clip)
        if args.decoder_coe > 0:
            loss_r.update(loss[0])
            loss_d.update(loss[1])
        else:
            loss_r.update(loss)
        scheduler.step()

        pred, gt = valid(valid_data, model, args.val_batch_size)
        # pred, gt = valid_mlp(valid_data, model, args.val_batch_size)
        rho, pval = stats.spearmanr(pred, gt)
        print('step %d, rho=%f, pval=%f' % (step, rho, pval))
        writer.add_scalar('rho', rho, step)
        writer.add_scalar('loss_r', loss_r.avg, step)
        writer.add_scalar('loss_d', loss_d.avg, step)
        if step % 10 == 0:
            fig_rank = plot_rank(pred, gt)
            writer.add_figure('rank_plot', fig_rank, step)
            plt.close(fig_rank)

            pred_gt = list(zip(pred, gt))
            pred_gt = sorted(pred_gt, key=lambda x: x[1])[:500]

            pred, gt = list(zip(*pred_gt))
            fig_value, ax = plt.subplots()
            ax.scatter(pred, gt, s=3)
            writer.add_figure('value_plot', fig_value, step)
            plt.close(fig_value)

            # print("=====pred_gt")
            # print(pred_gt[:10])

    model.save(os.path.join(logdir, 'model.pth'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--log_dir', type=str, default='logs/surrogate')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--tag', default=None, type=str)
    parser.add_argument('--model', default='gnn', type=str, choices=['gnn', 'mlp'])
    parser.add_argument('--dataset', default='101', type=str, choices=['101', '201'])

    args = parser.parse_args()
    main(args)
