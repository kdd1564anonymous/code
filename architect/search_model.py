##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##############################################################################
# Random Search and Reproducibility for Neural Architecture Search, UAI 2019 #
##############################################################################
import torch, random
import torch.nn as nn
from copy import deepcopy
from .cell_operations import ResNetBasicblock
from .search_cells import NAS201SearchCell as TinySearchCell, NASNetSearchCell as SearchCell
from .genotypes import Structure


class TinyNetwork(nn.Module):
    def __init__(self, C, N, max_nodes, num_classes, search_space, affine,
                 track_running_stats):
        super(TinyNetwork, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C))

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4
                                                            ] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [
            True
        ] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
                zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = TinySearchCell(C_prev, C_curr, 1, max_nodes,
                                      search_space, affine,
                                      track_running_stats)
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(
                        num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev),
                                     nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_cache = None

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells),
                                                       cell.extra_repr())
        return string

    def extra_repr(self):
        return (
            '{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.
            format(name=self.__class__.__name__, **self.__dict__))

    def random_genotype(self, set_cache):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name = random.choice(self.op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        arch = Structure(genotypes)
        if set_cache: self.arch_cache = arch
        return arch

    def set_genotype(self, genotype):
        self.arch_cache = genotype

    def forward(self, inputs):

        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, TinySearchCell):
                feature = cell.forward_dynamic(feature, self.arch_cache)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


class NASNetwork(nn.Module):
    def __init__(self, C, N, steps, multiplier, stem_multiplier, num_classes,
                 search_space, affine, track_running_stats, edges=2):
        super(NASNetwork, self).__init__()
        self._C = C
        self._layerN = N
        self._steps = steps
        self._multiplier = multiplier
        self._edges = edges
        self.stem = nn.Sequential(
            nn.Conv2d(3,
                      C * stem_multiplier,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(C * stem_multiplier))

        # config for each layer
        layer_channels = [C] * N + [C * 2] + [C * 2] * (N - 1) + [
            C * 4
        ] + [C * 4] * (N - 1)
        layer_reductions = [False] * N + [True] + [False] * (N - 1) + [
            True
        ] + [False] * (N - 1)

        num_edge, edge2index = None, None
        C_prev_prev, C_prev, C_curr, reduction_prev = C * stem_multiplier, C * stem_multiplier, C, False

        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
                zip(layer_channels, layer_reductions)):
            cell = SearchCell(search_space, steps, multiplier, C_prev_prev,
                              C_prev, C_curr, reduction, reduction_prev,
                              affine, track_running_stats)
            if num_edge is None:
                num_edge, edge2index = cell.num_edges, cell.edge2index
            else:
                assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(
                    num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev_prev, C_prev, reduction_prev = C_prev, multiplier * C_curr, reduction
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev),
                                     nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        # self.arch_normal_parameters = nn.Parameter(
        #     1e-3 * torch.randn(num_edge, len(search_space)))
        # self.arch_reduce_parameters = nn.Parameter(
        #     1e-3 * torch.randn(num_edge, len(search_space)))
        # self.tau = 10

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(
            self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells),
                                                       cell.extra_repr())
        return string

    def extra_repr(self):
        return (
            '{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})'
            .format(name=self.__class__.__name__, **self.__dict__))

    def forward(self, inputs):
        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                struct = self.arch_cache_reduce
            else:
                struct = self.arch_cache_normal
            s0, s1 = s1, cell.forward_dynamic(s0, s1, struct)
        out = self.lastact(s1)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits

    def _random_genotype(self):
        genotypes = []
        for i in range(self._steps):
            xlist = []
            if self._edges > 0:
                pre = random.sample(range(i + 2), self._edges)
            else:
                pre = range(i + 2)
            for j in pre:
                node_str = '{:}<-{:}'.format(i, j)
                if self._edges > 0:
                    op_name = random.choice(self.op_names[1:])
                else:
                    op_name = random.choice(self.op_names)

                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes, 2)

    def random_genotype(self, set_cache):
        arch_normal, arch_reduce = self._random_genotype(), self._random_genotype()
        if set_cache:
            self.set_genotype((arch_normal, arch_reduce))
        return arch_normal, arch_reduce

    def set_genotype(self, genos):
        normal, reduce = genos
        self.arch_cache_normal, self.arch_cache_reduce = normal, reduce

