import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import ECConv, GlobalAttention


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, t):
        super().__init__()
        self.pre = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.edge_nn = nn.Linear(t, in_channels * out_channels)
        self.ec = ECConv(in_channels, out_channels, self.edge_nn)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.pre(x)
        return self.ec(x, edge_index, edge_attr)


class BiGraph(nn.Module):
    def __init__(self, in_channels, out_channels, t):
        super().__init__()
        self.n = EdgeConv(in_channels, out_channels // 2, t)
        self.r = EdgeConv(in_channels, out_channels // 2, t)
    
    def forward(self, x, edge_index, edge_attr):
        edge_r = torch.flip(edge_index, dims=[-1])
        return torch.cat([
            self.n(x, edge_index, edge_attr),
            self.r(x, edge_r, edge_attr)], dim=-1)


class GraphRes(nn.Module):
    def __init__(self, *m):
        super().__init__()
        self.nn = nn.ModuleList(m)
    
    def forward(self, x, edge_index, edge_attr):
        for m in self.nn:
            x = x + m(x, edge_index, edge_attr)
        return x


class Encoder(nn.Module):
    def __init__(self, t_op: int, t_node: int, h_dim: int, n_layers=3):
        super().__init__()
        # self.node_init = nn.Embedding(t_node, h_dim)
        self.node_init = nn.Linear(t_node, h_dim, bias=False)

        self.conv = GraphRes(*[BiGraph(h_dim, h_dim, t_op) for _ in range(n_layers)])
        # self.conv = GraphRes(
        #     BiGraph(h_dim, h_dim, t_op),
        #     BiGraph(h_dim, h_dim, t_op),
        #     BiGraph(h_dim, h_dim, t_op)) 
        # self.conv = EdgeConv(h_dim, h_dim, t_op)
        # self.conv = BiGraph(h_dim, h_dim, t_op)
        # self.conv = GraphRes(
        #     EdgeConv(h_dim, h_dim, t_op),
        #     EdgeConv(h_dim, h_dim, t_op)) 


    def forward(self, x, edge_index, edge_attr):
        x = self.node_init(x)
        x = self.conv(x, edge_index, edge_attr)
        return x


class Decoder(nn.Module):
    def __init__(self, t_nodes: int, t_edges: int, h_dim: int):
        super().__init__()
        self.t_edges = t_edges
        self.t_nodes = t_nodes
        self.stem = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.ReLU(True)
            # nn.Dropout(0.2)
        )
        self.nn_node = nn.Sequential(
            nn.Linear(h_dim, t_nodes),
            nn.LayerNorm(t_nodes),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(t_nodes, t_nodes))

        self.nn_edge = nn.Sequential(
            nn.Linear(h_dim * 2, t_edges),
            nn.LayerNorm(t_edges),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(t_edges, t_edges))
    
    def forward(self, nodes, edges):
        nodes = self.stem(nodes)
        nodes_out = self.nn_node(nodes)
        edges_out = self.nn_edge(torch.cat([nodes[edges[0]], nodes[edges[1]]], dim=-1))
        return nodes_out, edges_out