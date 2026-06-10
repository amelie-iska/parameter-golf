"""Minimal TokenGT-style graph adapter for Parameter-Golf experiments.

This file deliberately avoids changing the baseline language-model path. It exposes a
small adapter that TropicalGT-I can import or copy into experiments when graph-token
conditioning is enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn


@dataclass
class TokenGTAdapterConfig:
    graph_feature_dim: int = 48
    model_dim: int = 384
    max_type_id: int = 4
    max_endpoint_id: int = 4096


class TokenGTGraphAdapter(nn.Module):
    def __init__(self, config: TokenGTAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_proj = nn.Sequential(nn.Linear(config.graph_feature_dim, config.model_dim), nn.GELU(), nn.LayerNorm(config.model_dim))
        self.type_emb = nn.Embedding(config.max_type_id, config.model_dim)
        # Endpoint ids let edge tokens carry TokenGT-style incidence structure.
        # A raw endpoint id of -1 maps to padding index 0; node ids map to id+1.
        self.endpoint_emb = nn.Embedding(config.max_endpoint_id + 1, config.model_dim)

    def forward(self, token_features: Tensor, token_type_ids: Tensor, mask: Tensor, endpoint_ids: Tensor | None = None) -> Tensor:
        type_ids = token_type_ids.clamp_min(0).clamp_max(self.config.max_type_id - 1)
        graph_tokens = self.feature_proj(token_features) + self.type_emb(type_ids)
        if endpoint_ids is not None:
            endpoints = endpoint_ids.clamp_min(-1).clamp_max(self.config.max_endpoint_id - 1) + 1
            endpoint_mask = endpoint_ids.ge(0).to(graph_tokens.dtype)[..., None]
            endpoint_context = (self.endpoint_emb(endpoints) * endpoint_mask).sum(dim=-2)
            endpoint_denom = endpoint_mask.sum(dim=-2).clamp_min(1.0)
            graph_tokens = graph_tokens + endpoint_context / endpoint_denom
        masked = graph_tokens * mask[..., None].to(graph_tokens.dtype)
        denom = mask.sum(dim=1).clamp_min(1).to(graph_tokens.dtype)[:, None]
        return masked.sum(dim=1) / denom


def graph_token_structural_bytes(mask: Tensor, token_type_ids: Tensor, endpoint_ids: Tensor | None = None, node_counts: Tensor | None = None) -> int:
    """Deterministic byte budget for TokenGT graph tokens.

    This mirrors the TropicalGT-I accounting used for graph-aware BPB.  It is a
    compact structural budget, not the explicit serialized JSON side information.
    """

    mask_cpu = mask.detach().cpu().bool()
    type_cpu = token_type_ids.detach().cpu()
    endpoints_cpu = endpoint_ids.detach().cpu() if endpoint_ids is not None else None
    node_counts_cpu = node_counts.detach().cpu() if node_counts is not None else None
    total = 0
    for row in range(mask_cpu.shape[0]):
        if node_counts_cpu is not None:
            node_count = max(int(node_counts_cpu[row].item()), 1)
        else:
            node_count = max(int(((type_cpu[row] == 0) & mask_cpu[row]).sum().item()), 1)
        endpoint_width = max(1, math.ceil(math.log2(node_count + 2) / 8.0))
        for col in range(mask_cpu.shape[1]):
            if not bool(mask_cpu[row, col].item()):
                continue
            total += 1
            if int(type_cpu[row, col].item()) == 1:
                if endpoints_cpu is None:
                    total += endpoint_width
                else:
                    valid = int(endpoints_cpu[row, col].ge(0).sum().item())
                    total += endpoint_width * max(valid, 1)
    return int(total)


def graph_bpb_metrics(
    nll: float | Tensor,
    target_bytes: int | Tensor,
    mask: Tensor,
    token_type_ids: Tensor,
    endpoint_ids: Tensor | None = None,
    explicit_graph_json_bytes: int | float = 0,
    graph_side_weight: float = 1.0,
    node_counts: Tensor | None = None,
) -> dict[str, float]:
    """Return text BPB plus TokenGT graph-aware BPB variants."""

    nll_value = float(nll.detach().cpu()) if torch.is_tensor(nll) else float(nll)
    byte_count = int(target_bytes.detach().cpu().item()) if torch.is_tensor(target_bytes) else int(target_bytes)
    graph_bytes = graph_token_structural_bytes(mask, token_type_ids, endpoint_ids=endpoint_ids, node_counts=node_counts)
    nll_bits = nll_value * max(byte_count, 1) / math.log(2.0)
    side_bits = float(graph_side_weight) * 8.0 * float(explicit_graph_json_bytes)
    return {
        "target_bytes": float(byte_count),
        "nll_bits": float(nll_bits),
        "bpb": float(nll_bits / max(byte_count, 1)),
        "text_bpb": float(nll_bits / max(byte_count, 1)),
        "graph_bpb": float((nll_bits + side_bits) / max(byte_count + graph_bytes, 1)),
        "graph_sideinfo_bpb": float((nll_bits + side_bits) / max(byte_count, 1)),
        "graph_conditioned_bpb_no_side_cost": float(nll_bits / max(byte_count + graph_bytes, 1)),
        "graph_token_structural_bytes": float(graph_bytes),
        "explicit_graph_json_bytes": float(explicit_graph_json_bytes),
        "graph_sideinfo_bits": float(side_bits),
    }
