import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch_geometric
import torch_geometric.data as gd
import math

from torch_geometric.nn import GlobalAttention, global_add_pool
from .encoder import Encoder, Decoder
from copy import deepcopy
from utils import AverageMeter, random_interpolate, grouper

from itertools import islice


class MLP(nn.Module):
    def __init__(self, t_op: int, n_pos: int, h_dim: int, n_out: int):
        super().__init__()
        self.inp = nn.Linear(t_op, h_dim)
        self.hidden = nn.Sequential(nn.LayerNorm(h_dim), nn.ReLU(True),
                                    nn.Linear(h_dim,
                                              h_dim), nn.LayerNorm(h_dim),
                                    nn.ReLU(True), nn.Linear(h_dim, h_dim),
                                    nn.LayerNorm(h_dim), nn.ReLU(True),
                                    nn.Linear(h_dim, h_dim))
        self.attn_a = nn.Linear(h_dim, 1)
        self.attn_v = nn.Sequential(nn.LayerNorm(h_dim), nn.ReLU(True),
                                    nn.Dropout(0.2), nn.Linear(h_dim, h_dim))
        self.out = nn.Sequential(nn.LayerNorm(h_dim), nn.ReLU(True),
                                 nn.Dropout(0.5), nn.Linear(h_dim, h_dim // 2),
                                 nn.LayerNorm(h_dim // 2), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(h_dim // 2, n_out))

    def graph_feature(self, x):
        xlist = x.to_data_list()
        x = torch.stack([v.edge_attr for v in xlist], dim=0)
        hs = self.inp(x)
        hs = self.hidden(hs)
        hs_a = F.softmax(self.attn_a(hs), dim=1)  # [batch, ops, 1]
        hs_v = self.attn_v(hs)  # [batch, ops, h_dim]
        features = torch.sum(hs_a * hs_v, dim=1)  # [batch, h_dim]
        return features

    def forward(self, x):
        return self.out(self.graph_feature(x))

    def criterion_rankpair(self, X, Y):
        pred = self(X)
        loss = []
        for i in range(1, Y.size(0)):
            for j in range(i):
                loss.append(rank_loss(Y[i], Y[j], pred[i], pred[j]))
        loss = sum(loss) / len(loss)
        return loss

    def rank_pair_step(self,
                       X,
                       optimizer: torch.optim.Optimizer,
                       grad_clip=5.):
        # Y [2, n_out]
        optimizer.zero_grad()
        loss = self.criterion_rankpair(X, X.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def fit(self,
            queue,
            optim,
            grad_clip=5.,
            decoder_coe=0.2):
        loss_r = AverageMeter()
        for batch in queue:
            if len(batch) < 2:
                break
            X = gd.Batch.from_data_list(batch).to('cuda')
            Y = X.y
            loss = self.rank_pair_step(X, optim, grad_clip)
            loss_r.update(loss)
        return loss_r.avg


class Predictor(nn.Module):
    def __init__(self,
                 t_edge: int,
                 t_node: int,
                 n_node: int,
                 h_dim: int,
                 n_out: int,
                 n_layers=3):
        super().__init__()
        self.t_edge = t_edge
        self.t_node = t_node
        self.n_node = n_node
        self.h_dim = h_dim
        self.n_out = n_out

        self.encoder = Encoder(t_edge, t_node, h_dim, n_layers)
        self.mean = nn.Linear(h_dim, h_dim)
        self.logvar = nn.Linear(h_dim, h_dim)
        self.decoder = Decoder(t_node, t_edge, h_dim)
        self.affine = nn.Sequential(nn.LayerNorm(h_dim), nn.ReLU(True),
                                    nn.Dropout(0.2), nn.Linear(h_dim, h_dim))
        self.gp = GlobalAttention(nn.Linear(h_dim, 1), self.affine)

        self.out = nn.Sequential(nn.LayerNorm(h_dim), nn.ReLU(True),
                                 nn.Dropout(0.5), nn.Linear(h_dim, h_dim // 2),
                                 nn.LayerNorm(h_dim // 2), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(h_dim // 2, n_out))

    def forward(self, x):
        f = self.graph_feature(x)
        return self.out(f)

    def graph_feature(self, x) -> torch.Tensor:
        f = self.encoder(x.x, x.edge_index, x.edge_attr)
        f = self.gp(f, x.batch)
        return f

    def criterion_rankpair(self, pred, target):
        loss = []
        for i in range(1, target.size(0)):
            for j in range(i):
                loss.append(rank_loss(target[i], target[j], pred[i], pred[j]))
        loss = sum(loss) / len(loss)
        return loss

    def rank_pair_step(self,
                       X,
                       optimizer: torch.optim.Optimizer,
                       grad_clip=5.,
                       n_out=-1,
                       decoder_coe=None):
        if n_out == -1:
            n_out = self.n_out
        # Y [2, n_out]
        optimizer.zero_grad()
        pred = self.forward(X)
        pred.view(X.y.size(0), -1)
        pred = pred[:, :n_out]
        loss_total = loss_r = self.criterion_rankpair(pred, X.y)

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()
        return loss_r.item()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def fit(self,
            queue,
            optim,
            grad_clip=5.,
            decoder_coe=0.2,
            norm=None):
        loss_r = AverageMeter()
        for batch in queue:
            if len(batch) < 2:
                break
            X = gd.Batch.from_data_list(batch).to('cuda')
            Y = X.y
            loss = self.rank_pair_step(X,
                                       optim,
                                       grad_clip,
                                       decoder_coe=decoder_coe)
            loss_r.update(loss)
        return loss_r.avg

    def grad_step_on_archs(self,
                           pool,
                           batch_size,
                           step_size,
                           checker):
        new_trace = []
        self.eval()
        last_idx = 0
        for archs in grouper(batch_size, pool):
            X = gd.Batch.from_data_list([a.binarized_data()
                                         for a in archs]).to('cuda')
            exp = self(X).view(-1).sum()  # + connective_regularisation
            exp.backward()
            new_archs = [a.step_grad(step_size) for a in archs]
            with torch.no_grad():
                new_X = gd.Batch.from_data_list(
                    [a.binarized_data() for a in new_archs]).to('cuda')
                new_exp = self(new_X).view(len(new_archs), -1)
                for a, e in zip(new_archs, new_exp):
                    if checker(a):
                        new_trace.append((a, e.item(), last_idx))
                    last_idx += 1
        return new_trace


def rank_loss(Yl, Yr, pl, pr):
    rl = -torch.log(torch.sigmoid((Yl - Yr).sign() * (pl - pr)))
    return rl.sum()
