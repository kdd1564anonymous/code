import torch_geometric
import torch_geometric.data as gd

import architect.genotypes as gt
import architect.cell_operations as cell
import numpy as np
import torch
import sys
from torch import tensor
from nasbench import api
from search.nas_101_utils import NASBench, spec2data
from tqdm import tqdm

data = api.NASBench(sys.argv[-1])
dataset = []

bench = NASBench(data)
for h in tqdm(bench._all_hashes):
    spec = bench.hash2spec(h)
    acc = bench.eval_arch(spec)
    dataset.append(spec2data(spec, acc))

train_length = len(dataset) // 2
val_length = len(dataset) - train_length
train_split, val_split = torch.utils.data.random_split(
    list(range(len(dataset))),
    [train_length, val_length])

valid_data = [dataset[idx] for idx in val_split]
train_data = [dataset[idx] for idx in train_split]


torch.save(valid_data, 'data/nas101-valid.pth')
torch.save(train_data, 'data/nas101-train.pth')
