# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import SGD
# from arch_generator import arch_generator
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import copy as cp
import torch_geometric as tg
import torch_geometric.data as gd
from collections import OrderedDict
# from keras.optimizers import Adam
import numpy
from random import shuffle

from utils import AverageMeter
from predictor import rank_loss
from itertools import islice


class AlphaxMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_dim = 56
        # self.env_model = self.build_env_model()
        self.env_model = nn.Sequential(nn.Linear(56, 512), nn.LayerNorm(512),
                                       nn.ReLU(True), nn.Dropout(0.2),
                                       nn.Linear(512, 2048),
                                       nn.LayerNorm(2048), nn.ReLU(True),
                                       nn.Dropout(0.2), nn.Linear(2048, 2048),
                                       nn.LayerNorm(2048), nn.ReLU(True),
                                       nn.Dropout(0.2), nn.Linear(2048, 512),
                                       nn.LayerNorm(512), nn.ReLU(True),
                                       nn.Dropout(0.2), nn.Linear(512, 1))

    def forward(self, batch):
        batch_list = batch.to_data_list()
        encode_list = [predict_encoder(datum) for datum in batch_list]
        X = torch.tensor(encode_list, dtype=torch.float, device=batch.x.device)
        return self.env_model(X)

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

    def fit(self, queue, optim, grad_clip=5., decoder_coe=0.2):
        loss_r = AverageMeter()
        for batch in queue:
            if len(batch) < 2:
                break
            X = gd.Batch.from_data_list(batch).to('cuda')
            Y = X.y
            loss = self.rank_pair_step(X, optim, grad_clip)
            loss_r.update(loss)
        return loss_r.avg



@torch.no_grad()
def net_encoder(net):
    net_code = list(net.x.argmax(dim=-1) + 2)
    while len(net_code) < 7:
        net_code.append(9)
    return net_code


def predict_encoder(network):

    network = network.to('cpu')
    net_arch = []
    net_code = net_encoder(network)
    network = tg.data.Batch.from_data_list([network])

    adj_mat = tg.utils.to_dense_adj(network.edge_index, network.batch,
                                    network.edge_attr)[0]
    for adj in adj_mat:
        for element in adj:
            net_arch.append(element)
    while len(net_arch) < 49:
        net_arch.append(0)
    net_arch += net_code
    return net_arch


def encoder(arch):
    concat_code = []
    for l, v in arch.items():
        net_info = []
        net_arch = []
        network = json.loads(l)
        node_list = network["node_list"]
        if node_list[-1] == 'term':
            node_list = node_list[:-1]
        adj_mat = network["adj_mat"]
        net_code = net_encoder(node_list)

        for adj in adj_mat:
            for element in adj:
                net_arch.append(element)
        while len(net_arch) < 49:
            net_arch.append(0)
        for code in net_code:
            net_arch.append(code)
        net_info.append(net_arch)
        net_info.append(v)
        concat_code.append(net_info)
    shuffle(concat_code)
    return concat_code
