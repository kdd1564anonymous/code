import torch_geometric
import torch_geometric.data as gd

import nas_201_api as nb
import architect.genotypes as gt
import architect.cell_operations as cell
import numpy as np
import torch
from torch import tensor

OP_IDX = {
    op: idx for idx, op in enumerate(cell.SearchSpaceNames['nas-bench-201'])
}


def idx2data(provider, idx: int):
    struct = gt.Structure.str2fullstructure(provider[idx])

    edge_index = [[], []]
    edge_attr = []
    nodes = [[1]] * (1 + len(struct))
    for node, ops in enumerate(struct):
        for op, pre in ops:
            edge_index[0].append(pre)
            edge_index[1].append(node + 1)
            edge_attr.append(torch.eye(len(OP_IDX))[OP_IDX[op]])
    
    x = gd.Data(
        x=tensor(nodes, dtype=torch.float),
        edge_index=tensor(edge_index, dtype=torch.long),
        edge_attr=torch.stack(edge_attr),
        y=tensor([idx2acc(provider, idx)])
    )
    return x


def idx2acc(provider, idx):
    return 100 - provider.get_more_info(idx, 'cifar10-valid', None, False, False)['valid-accuracy']

print('loading nas-bench-201')
nas_bench = nb.NASBench201API(sys.argv[-1])
duplicated = set()
dataset = []

for idx in range(len(nas_bench)):
    struct = gt.Structure.str2structure(nas_bench[idx])
    unique_str = struct.to_unique_str(True)
    if unique_str not in duplicated:
        duplicated.add(unique_str)
        dataset.append(idx2data(nas_bench, idx))

print("deduplicated archs: %d"%len(dataset))
train_length = len(dataset) // 2
val_length = len(dataset) - train_length
train_split, val_split = torch.utils.data.random_split(
    list(range(len(dataset))),
    [train_length, val_length])

valid_data = [dataset[idx] for idx in val_split]
# for d in valid_data:
#     print(d.y)
train_data = [dataset[idx] for idx in train_split]


torch.save(valid_data, 'data/nas201-valid.pth')
torch.save(train_data, 'data/nas201-train.pth')
