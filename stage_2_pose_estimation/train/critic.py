"""Diffusion score network for pairwise polygon alignment."""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from gnnFeature import PolygonGCN


class GaussianFourierProjection(nn.Module):
    """Embed scalar timesteps with fixed Gaussian Fourier features."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False,
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PolygonPackingTransformer(nn.Module):
    """Predict score vectors for two polygon actions under VE diffusion."""

    def __init__(
        self,
        marginal_prob_std_func,
        device,
        action_dim=4,
        feature_dim=64,
        hidden_dim=128,
        nhead=8,
        num_layers=4,
    ):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std_func

        # `device` is kept in the signature for checkpoint-loading compatibility.
        _ = device

        self.pair_gcn = PolygonGCN(out_dim=feature_dim)
        self.geom_proj = nn.Linear(feature_dim, hidden_dim)

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.t_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 4),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_head = nn.Linear(hidden_dim, 2)
        self.rot_head = nn.Linear(hidden_dim, 2)

    def forward(self, g1, g2, actions, t):
        node_feat_a, node_feat_b, _, _ = self.pair_gcn(g1, g2)

        dense_a, mask_a = to_dense_batch(node_feat_a, g1.batch)
        dense_b, mask_b = to_dense_batch(node_feat_b, g2.batch)
        dense_a = self.geom_proj(dense_a)
        dense_b = self.geom_proj(dense_b)

        time_params = self.t_embed(t)
        scale_a, shift_a, scale_b, shift_b = torch.chunk(time_params, 4, dim=-1)
        scale_a = scale_a.unsqueeze(1)
        shift_a = shift_a.unsqueeze(1)
        scale_b = scale_b.unsqueeze(1)
        shift_b = shift_b.unsqueeze(1)

        action_tokens = self.action_encoder(actions)
        action_a = action_tokens[:, 0:1, :]
        action_b = action_tokens[:, 1:2, :]

        feat_a = (dense_a + action_a) * (1 + scale_a) + shift_a
        feat_b = (dense_b + action_b) * (1 + scale_b) + shift_b

        tokens = torch.cat([feat_a, feat_b], dim=1)
        src_key_padding_mask = ~torch.cat([mask_a, mask_b], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)

        split_idx = dense_a.size(1)
        pooled_a = self.masked_mean_pool(encoded[:, :split_idx, :], mask_a)
        pooled_b = self.masked_mean_pool(encoded[:, split_idx:, :], mask_b)
        combined = torch.stack([pooled_a, pooled_b], dim=1)

        pos_score = self.pos_head(combined)
        rot_score = self.rot_head(combined)
        velocity = torch.cat([pos_score, rot_score], dim=-1)

        _, std = self.marginal_prob_std(torch.zeros_like(t), t)
        return velocity / std

    @staticmethod
    def masked_mean_pool(x, mask):
        """Average only valid nodes inside a dense batch."""
        x_masked = x * mask.unsqueeze(-1).float()
        num_nodes = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return x_masked.sum(dim=1) / num_nodes
