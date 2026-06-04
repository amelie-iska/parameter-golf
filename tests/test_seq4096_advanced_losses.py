from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "records"
    / "track_10min_16mb"
    / "2026-03-19_TrainingOptSeq4096"
    / "train_gpt.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("seq4096_train_gpt", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Args:
    advanced_loss_scale = 1.0
    graphcg_loss_weight = 0.05
    toric_tropical_loss_weight = 0.03
    slepian_loss_weight = 0.02
    koszul_bgg_loss_weight = 0.01
    analogy_loss_weight = 0.01
    advanced_loss_sample_tokens = 16
    toric_tropical_fan_bins = 8


def tiny_model(module):
    return module.GPT(
        vocab_size=32,
        num_layers=2,
        model_dim=16,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.02,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )


def test_advanced_embedding_losses_are_finite_and_backpropagate() -> None:
    module = load_module()
    model = tiny_model(module)
    input_ids = torch.arange(48, dtype=torch.long).reshape(2, 24) % 32

    loss, metrics = module.advanced_embedding_losses(model, Args, input_ids)

    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert metrics["advanced/graphcg_loss"].isfinite()
    assert metrics["advanced/toric_tropical_loss"].isfinite()
    assert metrics["advanced/slepian_pollak_loss"].isfinite()
    assert metrics["advanced/koszul_bgg_loss"].isfinite()
    assert metrics["advanced/analogy_loss"].isfinite()
    loss.backward()
    assert model.tok_emb.weight.grad is not None
    assert torch.isfinite(model.tok_emb.weight.grad).all()


def test_advanced_embedding_losses_can_be_disabled() -> None:
    module = load_module()
    model = tiny_model(module)
    input_ids = torch.arange(16, dtype=torch.long).reshape(1, 16) % 32

    class Disabled(Args):
        advanced_loss_scale = 0.0

    loss, metrics = module.advanced_embedding_losses(model, Disabled, input_ids)

    assert loss.item() == 0.0
    assert metrics == {}
