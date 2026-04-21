"""Graph encoder used by the diffusion score network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PolygonGCN(nn.Module):
    """Encode polygon node geometry and local patch features."""

    def __init__(self, out_dim=64):
        super().__init__()
        self.init_x = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(True),
        )
        self.init_f = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True),
        )

        self.conv1 = GCNConv(48, 32)
        self.conv2 = GCNConv(48 + 32, 32)
        self.conv3 = GCNConv(48 + 32 + 32, 32)
        self.conv4 = GCNConv(48 + 32 + 32 + 32, out_dim)

    def encode_single(self, graph):
        x0 = self.init_x(graph.x)
        f0 = self.init_f(graph.f)
        h = torch.cat([x0, f0], dim=1)

        h1 = F.relu(self.conv1(h, graph.edge_index))
        h2 = F.relu(self.conv2(torch.cat([h, h1], dim=1), graph.edge_index))
        h3 = F.relu(self.conv3(torch.cat([h, h1, h2], dim=1), graph.edge_index))
        h4 = F.relu(self.conv4(torch.cat([h, h1, h2, h3], dim=1), graph.edge_index))
        return h4, graph.g

    def forward(self, g1, g2):
        ha, ga = self.encode_single(g1)
        hb, gb = self.encode_single(g2)
        return ha, hb, ga, gb
