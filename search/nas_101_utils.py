import torch
import torch.nn.functional as F
import architect.genotypes as gt
import architect.cell_operations as cell
import torch_geometric.data as gd
import h5py
import numpy as np
import random

from nasbench import api
from torch import tensor
from typing import List, Tuple, Union, Optional
from copy import deepcopy

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS_FULL = [INPUT, CONV1X1, CONV3X3, MAXPOOL3X3, OUTPUT]
OPS = [CONV1X1, CONV3X3, MAXPOOL3X3]

N_NODES = 7
MAX_EDGES = 9
EDGE_INDEX = [[], []]  # [[from], [to]]
EDGE_MAP = []  # [(from, to)]
for i in range(1, N_NODES):
    for j in range(i):
        EDGE_INDEX[0].append(j)
        EDGE_INDEX[1].append(i)
        EDGE_MAP.append((j, i))
EDGE_INDEX = tensor(EDGE_INDEX).long()


class NASBench(object):
    def __init__(self, bench: api.NASBench):
        self.arch2acc = dict()
        # self.arch2loss = dict()
        self.nas_bench = bench
        self.nas_bench.reset_budget_counters()
        self._all_hashes = list(self.nas_bench.hash_iterator())
        self.history_spec = []
        self._history_data = []

    def eval_arch(self, model_spec: Union[api.ModelSpec, str]):
        if isinstance(model_spec, str):
            arch_str = model_spec
            model_spec = self.hash2spec(arch_str)
        else:
            arch_str = self.nas_bench._hash_spec(model_spec)
        if arch_str in self.arch2acc:
            return self.arch2acc[arch_str]
        try:
            acc = 1. - self.nas_bench.query(model_spec)['validation_accuracy']
        except TypeError:  # This is caused by unconnective of the graph. Penalize by 100.
            acc = 1.
        self.arch2acc[arch_str] = acc
        self.history_spec.append(model_spec)
        return acc

    def arch_str(self, model_spec: api.ModelSpec):
        if not model_spec.valid_spec:
            return None
        try:
            return self.nas_bench._hash_spec(model_spec)
        except Exception as e:
            print(model_spec.matrix)
            print(model_spec.ops)
            raise e

    def lookup(self, arch: Union[api.ModelSpec, str]):
        if isinstance(arch, str):
            arch_str = arch
        else:
            arch_str = self.nas_bench._hash_spec(arch)
        return self.arch2acc[arch_str]

    def valid_acc(self, arch: Union[api.ModelSpec, str]):
        return self.lookup(arch)  # return recorded valid acc

    def test_acc(self, arch: Union[api.ModelSpec, str]):
        if isinstance(arch, str):
            arch_str = arch
        else:
            arch_str = self.nas_bench._hash_spec(arch)
        _, metrics = self.nas_bench.get_metrics_from_hash(arch_str)
        return 1 - np.mean(
            [metrics[108][i]['final_test_accuracy']
             for i in range(3)])  # return mean test acc

    def choose_best(self, archs):
        return min(archs, key=lambda a: self.lookup(a))

    def history_data(self):
        return [spec2data(m, self.lookup(m)) for m in self.history_spec]

    def hash2spec(self, hash: str):
        meta = self.nas_bench.fixed_statistics[hash]
        return api.ModelSpec(matrix=meta['module_adjacency'],
                             ops=meta['module_operations'])
    
    def str2struct(self, s: str):
        return self.hash2spec(s)

    def random_spec(self):
        h = random.choice(self._all_hashes)
        return self.hash2spec(h)

    def log_arch(self, arch: str, step, name, logger, writer):
        valid_final_acc = self.valid_acc(arch)
        test_acc = self.test_acc(arch)
        spec = self.hash2spec(arch)

        logger.info('model spec:')
        logger.info(spec.matrix)
        logger.info(spec.ops)
        logger.info('valid err: %.4f' % valid_final_acc)
        logger.info('test err: %.4f' % test_acc)

        writer.add_scalar('%s/valid' % name, valid_final_acc, step)
        writer.add_scalar('%s/test' % name, test_acc, step)

    @property
    def total_cost(self):
        return self.nas_bench.get_budget_counters()[0]


def spec2data(model_spec: api.ModelSpec, acc: Optional[float] = None, padding=True):
    edge_attr = []
    for i in range(1, model_spec.matrix.shape[0]):
        for j in range(i):
            v = model_spec.matrix[j, i]
            edge_attr.append([v])

    node = []
    for op in model_spec.ops:
        node.append(
            torch.eye(len(OPS_FULL), dtype=torch.float)[OPS_FULL.index(op)])

    edge_index = [[], []]
    for i in range(1, len(node)):
        for j in range(i):
            edge_index[0].append(j)
            edge_index[1].append(i)

    x = gd.Data(x=torch.stack(node, dim=0),
                edge_index=tensor(edge_index, dtype=torch.long),
                edge_attr=tensor(edge_attr, dtype=torch.float))
    if acc is not None:
        x.y = tensor([acc])
    return x


class Architect(object):
    def __init__(self, device='cuda'):
        self.device = device
        self.edge_num = MAX_EDGES
        self.node_types = OPS
        self.nodes = torch.empty(
            [N_NODES - 2, len(self.node_types)],
            device=device,
            requires_grad=True)  # besides input and output nodes
        self.edge_index = EDGE_INDEX.to(device)
        self.edge_attr = torch.empty([self.edge_index.size(1), 1],
                                     device=device,
                                     requires_grad=True)

    @torch.no_grad()
    def randomize_(self):
        self.edge_attr.normal_().mul_(1e-3)
        self.nodes.normal_().mul_(1e-3)
        return self

    def set_edges_(self, edge_attr):
        self.edge_attr.data = edge_attr
        return self

    @torch.no_grad()
    def step_grad(self, step_size=1.):
        other = deepcopy(self)
        other.edge_attr.data.add_(-step_size, self.edge_attr.grad)
        self.edge_attr.grad.zero_()
        other.nodes.data.add_(-step_size, self.nodes.grad)
        self.nodes.grad.zero_()
        return other

    def binarized_data(self):
        # harden nodes
        node_probs = F.softmax(self.nodes, dim=-1)
        node_ids = node_probs.argmax(
            dim=-1) + 1  # 0 is input, add 1 to be op type
        nodes = torch.zeros([N_NODES, len(self.node_types) + 2],
                            device=self.device)
        nodes[1:-1].scatter_(-1, node_ids.view(-1, 1), 1)
        nodes[1:-1, 1:-1].add_(-node_probs.detach()).add_(node_probs)
        nodes[0, 0] = 1
        nodes[-1, -1] = 1

        # harden edges, prob > 0.5 and max to self.edge_num
        with torch.no_grad():
            edge_probs = self.edge_attr.sigmoid()
            edge_vals, edge_ids = edge_probs.view(-1).topk(self.edge_num)
            edge_attr = torch.zeros_like(self.edge_attr)
            edge_attr[edge_ids] = 1
            edge_attr.mul_(edge_probs > 0.5)
        for v, i in zip(edge_vals, edge_ids):
            if v > 0.5:
                edge_attr[i] = 1
        edge_softmax = F.softmax(self.edge_attr, dim=0)
        edge_attr.add_(-edge_softmax.detach()).add_(edge_softmax)

        return gd.Data(x=nodes,
                       edge_index=self.edge_index,
                       edge_attr=edge_attr)

    @property
    @torch.no_grad()
    def struct(self):
        node_ids = self.nodes.argmax(dim=-1)
        matrix = np.zeros([N_NODES, N_NODES], dtype=int)
        edge_vals, edge_ids = self.edge_attr.flatten().topk(self.edge_num)
        for v, i in zip(edge_vals, edge_ids):
            if v < 0.:
                continue
            matrix[EDGE_MAP[i]] = 1
        ops = [INPUT] + [self.node_types[i] for i in node_ids] + [OUTPUT]
        return api.ModelSpec(matrix=matrix, ops=ops)
