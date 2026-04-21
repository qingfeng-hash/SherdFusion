"""Global retrieval model for polygon fragment search."""

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
        x_branch = F.relu(self.node_lin(data.x))
        f_branch = F.relu(self.patch_fc1(data.f))
        f_branch = F.relu(self.patch_fc2(f_branch))
        g_branch = F.relu(self.global_fc1(data.g))
        g_branch = F.relu(self.global_fc2(g_branch))
        return torch.cat([x_branch, f_branch], dim=1), g_branch


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

    def forward(self, g1, g2, return_embedding=False):
        # `return_embedding` is kept for backward compatibility with old scripts.
        _ = return_embedding

        fa, ga = self.feature_emb(g1)
        fb, gb = self.feature_emb(g2)

        out_ha1 = self.feature_gath1(fa, g1.edge_index)
        ha1 = F.relu(self.proj1(torch.cat([out_ha1, fa], dim=1)) + out_ha1)
        out_hb1 = self.feature_gath1(fb, g2.edge_index)
        hb1 = F.relu(self.proj1(torch.cat([out_hb1, fb], dim=1)) + out_hb1)

        out_ha2 = self.feature_gath2(ha1, g1.edge_index)
        ha2 = F.relu(self.proj2(torch.cat([out_ha2, ha1], dim=1)) + out_ha2)
        out_hb2 = self.feature_gath2(hb1, g2.edge_index)
        hb2 = F.relu(self.proj2(torch.cat([out_hb2, hb1], dim=1)) + out_hb2)

        out_ha3 = self.feature_gath3(ha2, g1.edge_index)
        ha3 = F.relu(self.proj3(torch.cat([out_ha3, ha2], dim=1)) + out_ha3)
        out_hb3 = self.feature_gath3(hb2, g2.edge_index)
        hb3 = F.relu(self.proj3(torch.cat([out_hb3, hb2], dim=1)) + out_hb3)

        out_ha4 = self.feature_gath4(ha3, g1.edge_index)
        ha4 = F.relu(self.proj4(torch.cat([out_ha4, ha3], dim=1)) + out_ha4)
        out_hb4 = self.feature_gath4(hb3, g2.edge_index)
        hb4 = F.relu(self.proj4(torch.cat([out_hb4, hb3], dim=1)) + out_hb4)

        ga_proj = self.global_proj(ga)
        gb_proj = self.global_proj(gb)
        return ha4, hb4, ga_proj, gb_proj
