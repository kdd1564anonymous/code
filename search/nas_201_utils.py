import torch
import torch.nn.functional as F
import nas_201_api as nb
import architect.genotypes as gt
import architect.cell_operations as cell
import torch_geometric.data as gd

from torch import tensor
from typing import List, Tuple, Union, Optional
from copy import deepcopy

OP_SPACE = cell.SearchSpaceNames['nas-bench-201']
N_NODES = 4
OP_IDX = {op: idx for idx, op in enumerate(OP_SPACE)}


class CifarBench(object):
    def __init__(self, nas_bench):
        self.arch2acc = {}
        self.unique = set()
        self.nas_bench = nas_bench
        self.total_cost = 0.
        self._history_data = []

    def eval_arch(self, arch: str):
        if arch not in self.arch2acc:
            acc, cost = train_and_eval(arch,
                                       self.nas_bench,
                                       dataname='cifar10-valid',
                                       use_012_epoch_training=True)
            self.arch2acc[arch] = acc
            self.total_cost += cost
            self._history_data.append(arch2data(arch, acc))
            return acc, cost
        return self.arch2acc[arch], 0
    
    def lookup(self, arch):
        return self.arch2acc[arch]

    def valid_acc(self, arch: str):
        arch_idx = self.nas_bench.query_index_by_arch(arch)
        return 100 - self.nas_bench.get_more_info(
            arch_idx, 'cifar10-valid', None, False)['valid-accuracy']

    def test_acc(self, arch: str):
        arch_idx = self.nas_bench.query_index_by_arch(arch)
        return 100 - self.nas_bench.get_more_info(arch_idx, 'cifar10', None,
                                                  False)['test-accuracy']
    
    def history_data(self):
        # return [arch2data(a, ac) for a, ac in self.arch2acc.items()]
        return self._history_data
    
    def choose_best(self, arch):
        return min(arch, key=lambda a: self.arch2acc[a])
    
    def all_data(self, unique=True):
        all_arch_data = []
        duplicated = set()
        for idx in range(len(self.nas_bench)):
            struct = gt.Structure.str2structure(self.nas_bench[idx])
            unique_str = struct.to_unique_str(True)
            if unique and unique_str in duplicated:
                continue
            duplicated.add(unique_str)
            all_arch_data.append(
                arch2data(struct,
                        train_and_eval(idx, self.nas_bench, 'cifar10-valid', True)[0]))
        return all_arch_data
    
    def arch_str(self, arch):
        return arch.tostr()

    def log_arch(self, arch, step, name, logger, writer):
        valid_12_acc = self.lookup(arch)
        valid_final_acc = self.valid_acc(arch)
        test_acc = self.test_acc(arch)

        logger.info('\n' + self.nas_bench.query_by_arch(arch))

        writer.add_scalar('%s/valid12' % name, valid_12_acc, step)
        writer.add_scalar('%s/valid' % name, valid_final_acc, step)
        writer.add_scalar('%s/test' % name, test_acc, step)


class Architect(object):
    def __init__(self,
                 device='cuda'):
        self.n_nodes = N_NODES
        self.device = device
        self.edge_space = OP_SPACE
        self.nodes = torch.ones([self.n_nodes, 1], device=device)
        self.edge_index = create_graph(self.n_nodes).to(device)
        self.edge_attr = torch.empty(
            [self.edge_index.size(1), len(self.edge_space)],
            device=device,
            requires_grad=True)

    @torch.no_grad()
    def randomize_(self):
        self.edge_attr.normal_().mul_(1e-3)
        return self

    def set_edges_(self, edge_attr):
        self.edge_attr.data = edge_attr
        return self

    @torch.no_grad()
    def step_grad(self, step_size=1.):
        other = deepcopy(self)
        other.edge_attr.data.add_(-step_size, self.edge_attr.grad)
        self.edge_attr.grad.zero_()
        return other

    @torch.no_grad()
    def step_discrete(self, direction=-1):  # step for negtive grad
        max_grad_ids = torch.argmax(direction * self.edge_attr.grad, dim=-1)
        new_edges = torch.zeros_like(self.edge_attr)
        new_edges.scatter_(-1, max_grad_ids.view(-1, 1), 1)
        self.edge_attr.grad.zero_()
        return deepcopy(self).set_edges_(new_edges)

    def binarized_data(self):
        probs = F.softmax(self.edge_attr, dim=-1)
        ids = probs.argmax(-1)
        edge_attr = torch.zeros_like(probs)
        edge_attr.scatter_(-1, ids.view(-1, 1), 1)
        edge_attr = edge_attr - probs.detach() + probs
        return gd.Data(x=self.nodes,
                       edge_index=self.edge_index,
                       edge_attr=edge_attr)

    @property
    @torch.no_grad()
    def struct(self):
        genotype = []
        op_ids = torch.argmax(self.edge_attr, dim=-1)
        edge_idx = 0
        for i in range(1, self.n_nodes):
            xlist = []
            for j in range(i):
                op_name = self.edge_space[op_ids[edge_idx]]
                xlist.append((op_name, j))
                edge_idx += 1
            genotype.append(xlist)
        return gt.Structure(genotype)


def create_graph(n_nodes):
    edge_index = [[], []]
    for i in range(1, n_nodes):
        for j in range(i):
            edge_index[0].append(j)
            edge_index[1].append(i)

    edge_index = tensor(edge_index).long()

    return edge_index


# from https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py
def train_and_eval(arch: Union[str, int],
                   nas_bench,
                   dataname='cifar10-valid',
                   use_012_epoch_training=True):
    if isinstance(arch, int):
        arch_index = arch
    else:
        arch_index = nas_bench.query_index_by_arch(arch)
    if use_012_epoch_training and nas_bench is not None:
        assert arch_index >= 0, 'can not find this arch : {:}'.format(arch)
        info = nas_bench.get_more_info(arch_index, dataname, None, True)
        valid_acc, time_cost = info[
            'valid-accuracy'], info['train-all-time'] + info['valid-per-time']
    elif not use_012_epoch_training and nas_bench is not None:
        nepoch = 25
        assert arch_index >= 0, 'can not find this arch : {:}'.format(arch)
        xoinfo = nas_bench.get_more_info(arch_index, 'cifar10-valid', None,
                                         True)
        xocost = nas_bench.get_cost_info(arch_index, 'cifar10-valid', False)
        info = nas_bench.get_more_info(arch_index, dataname, nepoch, False,
                                       True)
        cost = nas_bench.get_cost_info(arch_index, dataname, False)
        nums = {
            'ImageNet16-120-train': 151700,
            'ImageNet16-120-valid': 3000,
            'cifar10-valid-train': 25000,
            'cifar10-valid-valid': 25000,
            'cifar100-train': 50000,
            'cifar100-valid': 5000
        }
        estimated_train_cost = xoinfo['train-per-time'] / nums[
            'cifar10-valid-train'] * nums['{:}-train'.format(
                dataname)] / xocost['latency'] * cost['latency'] * nepoch
        estimated_valid_cost = xoinfo['valid-per-time'] / nums[
            'cifar10-valid-valid'] * nums['{:}-valid'.format(
                dataname)] / xocost['latency'] * cost['latency']
        try:
            valid_acc, time_cost = info[
                'valid-accuracy'], estimated_train_cost + estimated_valid_cost
        except:
            valid_acc, time_cost = info[
                'valtest-accuracy'], estimated_train_cost + estimated_valid_cost
    else:
        raise ValueError('NOT IMPLEMENT YET')
    obj = 100 - valid_acc
    return obj, time_cost


def arch2data(arch: Union[str, gt.Structure], acc=None):
    if isinstance(arch, str):
        struct = gt.Structure.str2fullstructure(arch)
    else:
        struct = arch
    edge_index = [[], []]
    edge_attr = []
    nodes = [[1.] for _ in range(1 + len(struct))]
    for idx, ops in enumerate(struct):
        for op, pre in ops:
            edge_index[0].append(pre)
            edge_index[1].append(idx + 1)
            edge_attr.append(torch.eye(len(OP_IDX))[OP_IDX[op]])

    x = gd.Data(x=tensor(nodes),
                edge_index=tensor(edge_index, dtype=torch.long),
                edge_attr=torch.stack(edge_attr))
    if acc is not None:
        x.y = tensor([acc])

    return x


def arch2acc(provider, arch, dataname='cifar10-valid', use12epoch=True):
    if isinstance(arch, int):
        arch_index = arch
    else:
        arch_index = provider.query_index_by_arch(arch)
    assert arch_index >= 0, 'can not find this arch : {:}'.format(arch)
    info = provider.get_more_info(arch_index, dataname, None, use12epoch)
    valid_acc = info['valid-accuracy']

    return 100 - valid_acc


def edge2arch(edge_attr):
    genotype = []
    op_ids = torch.argmax(edge_attr, dim=-1)
    edge_idx = 0
    for i in range(1, N_NODES):
        xlist = []
        for j in range(i):
            op_name = OP_SPACE[op_ids[edge_idx]]
            xlist.append((op_name, j))
            edge_idx += 1
        genotype.append(xlist)
    return gt.Structure(genotype).tostr()
