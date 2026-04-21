"""Retrieval model for polygon fragment matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FeatureEmbedding(nn.Module):
    """Embed node coordinates, local patch features, and global descriptors."""

    def __init__(
        self,
        node_feat_dim=3,
        patch_feat_dim=128,
        global_feat_dim=128,
        hidden_dim=128,
    ):
        super().__init__()
        self.node_lin = nn.Linear(node_feat_dim, hidden_dim)
        self.patch_fc1 = nn.Linear(patch_feat_dim, 256)
        self.patch_fc2 = nn.Linear(256, hidden_dim)
        self.global_fc1 = nn.Linear(global_feat_dim, 256)
        self.global_fc2 = nn.Linear(256, hidden_dim)

    def forward(self, data):
        node_branch = F.relu(self.node_lin(data.x))
        patch_branch = F.relu(self.patch_fc1(data.f))
        patch_branch = F.relu(self.patch_fc2(patch_branch))
        global_branch = F.relu(self.global_fc1(data.g))
        global_branch = F.relu(self.global_fc2(global_branch))
        return torch.cat([node_branch, patch_branch], dim=1), global_branch


class PolygonFeatureGathering(nn.Module):
    """A single graph-convolution block followed by projection."""

    def __init__(self, in_dim, hidden_dim=128, out_dim=64):
        super().__init__()
        self.gcn = GCNConv(in_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, h, edge_index):
        h = F.relu(self.gcn(h, edge_index))
        return self.fc(h)


class PolygonMatchingNet(nn.Module):
    """Encode polygons into local and global descriptors for retrieval."""

    def __init__(
        self,
        node_feat_dim=3,
        patch_feat_dim=128,
        global_feat_dim=128,
        hidden_dim=128,
        out_dim=64,
    ):
        super().__init__()
        self.feature_emb = FeatureEmbedding(
            node_feat_dim,
            patch_feat_dim,
            global_feat_dim,
            hidden_dim,
        )

        self.feature_gath1 = PolygonFeatureGathering(hidden_dim * 2, hidden_dim, out_dim)
        self.feature_gath2 = PolygonFeatureGathering(out_dim, hidden_dim, out_dim)
        self.feature_gath3 = PolygonFeatureGathering(out_dim, hidden_dim, out_dim)
        self.feature_gath4 = PolygonFeatureGathering(out_dim, hidden_dim, out_dim)

        self.proj1 = nn.Linear(out_dim + hidden_dim * 2, out_dim)
        self.proj2 = nn.Linear(out_dim + out_dim, out_dim)
        self.proj3 = nn.Linear(out_dim + out_dim, out_dim)
        self.proj4 = nn.Linear(out_dim + out_dim, out_dim)
        self.global_proj = nn.Linear(hidden_dim, out_dim)

    def encode_graph(self, graph_batch):
        """Encode one PyG batch into node embeddings and one global embedding."""
        features, global_features = self.feature_emb(graph_batch)

        out1 = self.feature_gath1(features, graph_batch.edge_index)
        h1 = F.relu(self.proj1(torch.cat([out1, features], dim=1)) + out1)

        out2 = self.feature_gath2(h1, graph_batch.edge_index)
        h2 = F.relu(self.proj2(torch.cat([out2, h1], dim=1)) + out2)

        out3 = self.feature_gath3(h2, graph_batch.edge_index)
        h3 = F.relu(self.proj3(torch.cat([out3, h2], dim=1)) + out3)

        out4 = self.feature_gath4(h3, graph_batch.edge_index)
        h4 = F.relu(self.proj4(torch.cat([out4, h3], dim=1)) + out4)

        return h4, self.global_proj(global_features)

    def forward(self, g1, g2):
        feat_a, global_a = self.encode_graph(g1)
        feat_b, global_b = self.encode_graph(g2)
        return feat_a, feat_b, global_a, global_b
