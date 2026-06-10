"""Minimal TokenGT-style graph adapter for Parameter-Golf experiments.

This file deliberately avoids changing the baseline language-model path. It exposes a
small adapter that TropicalGT-I can import or copy into experiments when graph-token
conditioning is enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn


@dataclass
class TokenGTAdapterConfig:
    graph_feature_dim: int = 48
    model_dim: int = 384
    max_type_id: int = 4


class TokenGTGraphAdapter(nn.Module):
    def __init__(self, config: TokenGTAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_proj = nn.Sequential(nn.Linear(config.graph_feature_dim, config.model_dim), nn.GELU(), nn.LayerNorm(config.model_dim))
        self.type_emb = nn.Embedding(config.max_type_id, config.model_dim)

    def forward(self, token_features: Tensor, token_type_ids: Tensor, mask: Tensor) -> Tensor:
        type_ids = token_type_ids.clamp_min(0).clamp_max(self.config.max_type_id - 1)
        graph_tokens = self.feature_proj(token_features) + self.type_emb(type_ids)
        masked = graph_tokens * mask[..., None].to(graph_tokens.dtype)
        denom = mask.sum(dim=1).clamp_min(1).to(graph_tokens.dtype)[:, None]
        return masked.sum(dim=1) / denom
