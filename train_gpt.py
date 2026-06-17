"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # First-class FineWeb graphification.  This is intentionally separate from
    # the ToricGT sidecar: the main BPB model itself receives TokenGT-style
    # graph-structural features by default.  Disable with FINEWEB_GRAPHIFY=0.
    fineweb_graphify = bool(int(os.environ.get("FINEWEB_GRAPHIFY", "1")))
    tokengt_first_class = bool(
        int(os.environ.get("TOKENGT_FIRST_CLASS", "1" if fineweb_graphify else "0"))
    ) and fineweb_graphify
    tokengt_first_class_lr = float(os.environ.get("TOKENGT_FIRST_CLASS_LR", 2e-4))
    tokengt_graph_radius = int(os.environ.get("TOKENGT_GRAPH_RADIUS", 4))
    tokengt_token_class_buckets = int(os.environ.get("TOKENGT_TOKEN_CLASS_BUCKETS", 64))
    tokengt_position_buckets = int(os.environ.get("TOKENGT_POSITION_BUCKETS", 256))
    tokengt_structural_weight = float(os.environ.get("TOKENGT_STRUCTURAL_WEIGHT", 0.050))
    tokengt_edge_weight = float(os.environ.get("TOKENGT_EDGE_WEIGHT", 0.035))
    tokengt_torus_weight = float(os.environ.get("TOKENGT_TORUS_WEIGHT", 0.015))
    tokengt_identifier_dim = int(os.environ.get("TOKENGT_IDENTIFIER_DIM", 24))
    tokengt_identifier_weight = float(os.environ.get("TOKENGT_IDENTIFIER_WEIGHT", 0.014))
    tokengt_endpoint_weight = float(os.environ.get("TOKENGT_ENDPOINT_WEIGHT", 0.018))
    tokengt_edge_token_weight = float(os.environ.get("TOKENGT_EDGE_TOKEN_WEIGHT", 0.016))
    graph_output_flattening = bool(
        int(
            os.environ.get(
                "OAI_FINEWEB_OUTPUT_FLATTENING",
                os.environ.get("GRAPH_OUTPUT_FLATTENING", "1" if fineweb_graphify else "0"),
            )
        )
    )
    graph_output_flattening_lr = float(os.environ.get("GRAPH_OUTPUT_FLATTENING_LR", tokengt_first_class_lr))
    graph_output_edge_radius = int(os.environ.get("GRAPH_OUTPUT_EDGE_RADIUS", tokengt_graph_radius))
    graph_output_node_weight = float(os.environ.get("GRAPH_OUTPUT_NODE_WEIGHT", 0.05))
    graph_output_edge_weight = float(os.environ.get("GRAPH_OUTPUT_EDGE_WEIGHT", 0.05))
    graph_output_virtual_edge_tokens = bool(int(os.environ.get("GRAPH_OUTPUT_VIRTUAL_EDGE_TOKENS", "1")))
    graph_output_edge_token_weight = float(os.environ.get("GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT", 0.035))
    graph_output_score_correction = bool(int(os.environ.get("GRAPH_OUTPUT_SCORE_CORRECTION", "1")))
    graph_output_score_correction_weight = float(os.environ.get("GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT", 0.024))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Bonafide graph LM stream plus ToricGT sidecar.  FineWeb remains the
    # score-defining BPB stream; graph Parquet rows are also graphified into
    # autoregressive node/edge token records and trained directly on the base
    # LM with a small primary-mixture weight.
    toricgt_sidecar = bool(int(os.environ.get("TORICGT_SIDECAR", "0")))
    require_toricgt_sidecar = bool(int(os.environ.get("REQUIRE_TORICGT_SIDECAR", "0")))
    graph_data_path = os.environ.get(
        "GRAPH_DATA_PATH",
        "/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/curated_hf_shards",
    )
    graph_train_glob = os.environ.get("GRAPH_TRAIN_GLOB", os.path.join(graph_data_path, "train", "*.parquet"))
    graph_lm_primary = bool(int(os.environ.get("GRAPH_LM_PRIMARY", "1" if toricgt_sidecar else "0")))
    require_graph_lm_primary = bool(int(os.environ.get("REQUIRE_GRAPH_LM_PRIMARY", "0")))
    graph_lm_seq_len = int(os.environ.get("GRAPH_LM_SEQ_LEN", train_seq_len))
    graph_lm_batch_size = int(os.environ.get("GRAPH_LM_BATCH_SIZE", 2))
    graph_lm_every = int(os.environ.get("GRAPH_LM_EVERY", 1))
    graph_lm_loss_weight = float(os.environ.get("GRAPH_LM_LOSS_WEIGHT", 0.08))
    graph_lm_loss_weight_start = float(
        os.environ.get("GRAPH_LM_LOSS_WEIGHT_START", min(0.025, graph_lm_loss_weight))
    )
    graph_lm_warmup_steps = int(os.environ.get("GRAPH_LM_WARMUP_STEPS", 350))
    graph_lm_hold_steps = int(os.environ.get("GRAPH_LM_HOLD_STEPS", 25))
    aux_grad_routing = bool(int(os.environ.get("AUX_GRAD_ROUTING", "1")))
    aux_grad_route_graph_lm = bool(int(os.environ.get("AUX_GRAD_ROUTE_GRAPH_LM", "1")))
    aux_grad_route_sidecar = bool(int(os.environ.get("AUX_GRAD_ROUTE_SIDECAR", "1")))
    aux_grad_route_teacher = bool(int(os.environ.get("AUX_GRAD_ROUTE_TEACHER", "1")))
    aux_grad_conflict_projection = bool(int(os.environ.get("AUX_GRAD_CONFLICT_PROJECTION", "1")))
    aux_grad_aligned_boost = float(os.environ.get("AUX_GRAD_ALIGNED_BOOST", 1.0))
    teacher_checkpoint = os.environ.get("TEACHER_CHECKPOINT", "")
    teacher_distill_weight = float(os.environ.get("TEACHER_DISTILL_WEIGHT", 0.0))
    teacher_distill_weight_start = float(os.environ.get("TEACHER_DISTILL_WEIGHT_START", teacher_distill_weight))
    teacher_distill_until_step = int(os.environ.get("TEACHER_DISTILL_UNTIL_STEP", 0))
    teacher_distill_decay_steps = int(os.environ.get("TEACHER_DISTILL_DECAY_STEPS", 0))
    teacher_distill_temperature = float(os.environ.get("TEACHER_DISTILL_TEMPERATURE", 2.0))
    teacher_distill_max_sequences = int(os.environ.get("TEACHER_DISTILL_MAX_SEQUENCES", 2))
    fineweb_caseops = bool(int(os.environ.get("FINEWEB_CASEOPS", "0")))
    score_first_tta = bool(int(os.environ.get("SCORE_FIRST_TTA", "0")))
    score_first_tta_steps = int(os.environ.get("SCORE_FIRST_TTA_STEPS", 0))
    score_first_tta_lr = float(os.environ.get("SCORE_FIRST_TTA_LR", 0.0))
    sidecar_seq_len = int(os.environ.get("TORICGT_SIDECAR_SEQ_LEN", min(256, train_seq_len)))
    sidecar_batch_size = int(os.environ.get("TORICGT_SIDECAR_BATCH_SIZE", 8))
    sidecar_every = int(os.environ.get("TORICGT_SIDECAR_EVERY", 1))
    sidecar_lr = float(os.environ.get("TORICGT_SIDECAR_LR", 2e-4))
    sidecar_loss_weight = float(os.environ.get("TORICGT_SIDECAR_LOSS_WEIGHT", 1.0))
    graphcg_loss_weight = float(os.environ.get("GRAPHCG_LOSS_WEIGHT", 2e-5))
    analogy_loss_weight = float(os.environ.get("ANALOGY_LOSS_WEIGHT", 2e-5))
    tokengt_graph_loss_weight = float(os.environ.get("TOKENGT_GRAPH_LOSS_WEIGHT", 4e-5))
    trajectory_memory_loss_weight = float(os.environ.get("TRAJECTORY_MEMORY_LOSS_WEIGHT", 2e-5))
    sidecar_compute_all_metrics = int(os.environ.get("TORICGT_SIDECAR_COMPUTE_ALL_METRICS", "1"))
    toric_geometry_loss_weight = float(os.environ.get("TORIC_GEOMETRY_LOSS_WEIGHT", 0.0))
    toric_vector_bundle_loss_weight = float(os.environ.get("TORIC_VECTOR_BUNDLE_LOSS_WEIGHT", 0.0))
    toric_bgg_loss_weight = float(os.environ.get("TORIC_BGG_LOSS_WEIGHT", 0.0))
    koszul_persistence_loss_weight = float(os.environ.get("KOSZUL_PERSISTENCE_LOSS_WEIGHT", 0.0))
    combinatorial_toric_loss_weight = float(os.environ.get("COMBINATORIAL_TORIC_LOSS_WEIGHT", 0.0))
    toric_geometry_num_exponents = int(os.environ.get("TORIC_GEOMETRY_NUM_EXPONENTS", 16))
    toric_geometry_exponent_dim = int(os.environ.get("TORIC_GEOMETRY_EXPONENT_DIM", 4))
    toric_geometry_probe_rank = int(os.environ.get("TORIC_GEOMETRY_PROBE_RANK", 12))
    toric_geometry_quant_bits = int(os.environ.get("TORIC_GEOMETRY_QUANT_BITS", 6))
    toric_geometry_max_positions = int(os.environ.get("TORIC_GEOMETRY_MAX_POSITIONS", 128))
    toric_vector_bundle_rank = int(os.environ.get("TORIC_VECTOR_BUNDLE_RANK", 8))
    toric_vector_bundle_num_one_dimensional_cones = int(
        os.environ.get("TORIC_VECTOR_BUNDLE_NUM_ONE_DIMENSIONAL_CONES", 8)
    )
    toric_vector_bundle_num_cones = int(os.environ.get("TORIC_VECTOR_BUNDLE_NUM_CONES", 8))
    toric_vector_bundle_filtration_levels = int(os.environ.get("TORIC_VECTOR_BUNDLE_FILTRATION_LEVELS", 3))
    toric_vector_bundle_max_positions = int(os.environ.get("TORIC_VECTOR_BUNDLE_MAX_POSITIONS", 96))
    toric_bgg_num_standard_tokens = int(os.environ.get("TORIC_BGG_NUM_STANDARD_TOKENS", 8))
    toric_bgg_probe_rank = int(os.environ.get("TORIC_BGG_PROBE_RANK", 8))
    toric_bgg_signature_dim = int(os.environ.get("TORIC_BGG_SIGNATURE_DIM", 16))
    toric_bgg_max_positions = int(os.environ.get("TORIC_BGG_MAX_POSITIONS", 64))
    koszul_max_points = int(os.environ.get("KOSZUL_MAX_POINTS", 16))
    koszul_max_windows = int(os.environ.get("KOSZUL_MAX_WINDOWS", 2))
    koszul_window_size = int(os.environ.get("KOSZUL_WINDOW_SIZE", 32))
    koszul_step_stride = int(os.environ.get("KOSZUL_STEP_STRIDE", 16))
    koszul_num_parameters = int(os.environ.get("KOSZUL_NUM_PARAMETERS", 3))
    koszul_temperature = float(os.environ.get("KOSZUL_TEMPERATURE", 0.12))
    koszul_chart_exponents = int(os.environ.get("KOSZUL_CHART_EXPONENTS", 10))
    combinatorial_toric_max_points = int(os.environ.get("COMBINATORIAL_TORIC_MAX_POINTS", 16))
    combinatorial_toric_max_windows = int(os.environ.get("COMBINATORIAL_TORIC_MAX_WINDOWS", 2))
    combinatorial_toric_window_size = int(os.environ.get("COMBINATORIAL_TORIC_WINDOW_SIZE", 32))
    combinatorial_toric_step_stride = int(os.environ.get("COMBINATORIAL_TORIC_STEP_STRIDE", 16))
    combinatorial_toric_num_chambers = int(os.environ.get("COMBINATORIAL_TORIC_NUM_CHAMBERS", 8))
    combinatorial_toric_exponent_dim = int(os.environ.get("COMBINATORIAL_TORIC_EXPONENT_DIM", 4))
    retrieval_conditioned_aux = bool(int(os.environ.get("RETRIEVAL_CONDITIONED_AUX", "1")))
    retrieval_gate_min = float(os.environ.get("RETRIEVAL_GATE_MIN", 0.20))
    retrieval_gate_softness = float(os.environ.get("RETRIEVAL_GATE_SOFTNESS", 0.08))
    retrieval_gate_center = float(os.environ.get("RETRIEVAL_GATE_CENTER", 0.20))
    sidecar_uncertainty_weighting = bool(int(os.environ.get("SIDECAR_UNCERTAINTY_WEIGHTING", "1")))
    sidecar_uncertainty_alpha = float(os.environ.get("SIDECAR_UNCERTAINTY_ALPHA", 0.35))
    sidecar_uncertainty_center = float(os.environ.get("SIDECAR_UNCERTAINTY_CENTER", 2.6))
    sidecar_uncertainty_scale = float(os.environ.get("SIDECAR_UNCERTAINTY_SCALE", 1.0))
    sidecar_uncertainty_max = float(os.environ.get("SIDECAR_UNCERTAINTY_MAX", 2.0))
    graphcg_bpb_orthogonal_weight = float(os.environ.get("GRAPHCG_BPB_ORTHOGONAL_WEIGHT", 0.02))
    graphcg_covariance_conflict_damping = float(os.environ.get("GRAPHCG_COVARIANCE_CONFLICT_DAMPING", 0.50))
    cas_toric_ideal_certificate_path = os.environ.get("CAS_TORIC_IDEAL_CERTIFICATE_PATH", "")
    checkpoint_every = int(os.environ.get("CHECKPOINT_EVERY", 0))
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints/parameter_golf_oai_dense")
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", "")
    start_step = int(os.environ.get("START_STEP", "0"))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def build_caseops_token_map(sp: spm.SentencePieceProcessor, vocab_size: int) -> Tensor:
    """Build an involutive, byte-length-preserving ASCII case token map."""

    token_map = torch.arange(vocab_size, dtype=torch.long)
    by_key: dict[tuple[bool, str], int] = {}
    meta: dict[int, tuple[bool, str, int]] = {}
    for token_id in range(min(vocab_size, int(sp.vocab_size()))):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id) or sp.is_byte(token_id):
            continue
        piece = sp.id_to_piece(token_id)
        leading = piece.startswith("▁")
        visible = piece[1:] if leading else piece
        if not visible or visible.swapcase() == visible:
            continue
        byte_len = len(visible.encode("utf-8"))
        by_key[(leading, visible)] = token_id
        meta[token_id] = (leading, visible, byte_len)
    for token_id, (leading, visible, byte_len) in meta.items():
        partner = by_key.get((leading, visible.swapcase()))
        if partner is None:
            continue
        _, partner_visible, partner_byte_len = meta.get(partner, (leading, "", -1))
        if partner_visible.swapcase() == visible and partner_byte_len == byte_len:
            token_map[token_id] = partner
            token_map[partner] = token_id
    return token_map


def apply_token_map(tokens: Tensor, token_map: Tensor | None) -> Tensor:
    if token_map is None:
        return tokens
    return token_map[tokens.to(dtype=torch.long)].to(dtype=tokens.dtype)


def load_validation_tokens(pattern: str, seq_len: int, token_map: Tensor | None = None) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    tokens = apply_token_map(tokens, token_map)
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    tta_optimizer = None
    tta_restore: list[tuple[nn.Parameter, bool]] = []
    tta_steps_done = 0
    if args.score_first_tta:
        eval_module = model.module if hasattr(model, "module") else model
        eval_base = getattr(eval_module, "_orig_mod", eval_module)
        tta_params: list[nn.Parameter] = []
        for attr in ("graph_output_flattening", "fineweb_tokengt"):
            module = getattr(eval_base, attr, None)
            if module is not None:
                tta_params.extend(list(module.parameters()))
        if tta_params:
            tta_param_ids = {id(param) for param in tta_params}
            for param in eval_base.parameters():
                tta_restore.append((param, bool(param.requires_grad)))
                param.requires_grad_(id(param) in tta_param_ids)
            tta_optimizer = torch.optim.AdamW(tta_params, lr=max(float(args.score_first_tta_lr), 1e-8))
    try:
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
            if tta_optimizer is not None and tta_steps_done < max(int(args.score_first_tta_steps), 0):
                tta_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    tta_loss = model(x, y)
                tta_loss.backward()
                tta_optimizer.step()
                tta_optimizer.zero_grad(set_to_none=True)
                tta_steps_done += 1
    finally:
        for param, requires_grad in tta_restore:
            param.requires_grad_(requires_grad)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def count_sentencepiece_target_bytes(
    previous_ids: Tensor,
    target_ids: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> Tensor:
    """Count decoded target bytes with the same normalization as validation BPB."""

    prev_flat = previous_ids.reshape(-1)
    tgt_flat = target_ids.reshape(-1)
    token_bytes = base_bytes_lut[tgt_flat].to(dtype=torch.int16)
    token_bytes += (has_leading_space_lut[tgt_flat] & ~is_boundary_token_lut[prev_flat]).to(dtype=torch.int16)
    return token_bytes.to(torch.float64).sum()

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str, token_map: Tensor | None = None):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.token_map = token_map
        self.file_idx = 0
        self.tokens = apply_token_map(load_data_shard(self.files[0]), self.token_map)
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = apply_token_map(load_data_shard(self.files[self.file_idx]), self.token_map)
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        token_map: Tensor | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern, token_map=token_map)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


from toricgt.oai_sidecar import GraphParquetTokenStream, ToricGTSidecar

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class FineWebTokenGTStructuralEmbedding(nn.Module):
    """Causal TokenGT-style graph features for the main FineWeb LM stream.

    Each SP1024 token is a node in a graph.  Directed one-dimensional local
    edges point from already revealed previous tokens into the current token.
    The module emits a compact structural embedding, so graphification is part
    of the primary BPB model instead of an auxiliary sidecar-only loss.
    """

    def __init__(
        self,
        dim: int,
        *,
        token_class_buckets: int,
        position_buckets: int,
        edge_radius: int,
        structural_weight: float,
        edge_weight: float,
        torus_weight: float,
        identifier_dim: int,
        identifier_weight: float,
        endpoint_weight: float,
        edge_token_weight: float,
        theta: float = 0.6180339887498948,
    ) -> None:
        super().__init__()
        if token_class_buckets <= 0:
            raise ValueError("TOKENGT_TOKEN_CLASS_BUCKETS must be positive")
        if position_buckets <= 0:
            raise ValueError("TOKENGT_POSITION_BUCKETS must be positive")
        if edge_radius < 0:
            raise ValueError("TOKENGT_GRAPH_RADIUS must be nonnegative")
        if identifier_dim <= 0:
            raise ValueError("TOKENGT_IDENTIFIER_DIM must be positive")
        self.dim = int(dim)
        self.token_class_buckets = int(token_class_buckets)
        self.position_buckets = int(position_buckets)
        self.edge_radius = int(edge_radius)
        self.structural_weight = float(structural_weight)
        self.edge_weight = float(edge_weight)
        self.torus_weight = float(torus_weight)
        self.identifier_half_dim = max(1, int(math.ceil(identifier_dim / 2)))
        self.identifier_feature_dim = 2 * self.identifier_half_dim
        self.identifier_weight = float(identifier_weight)
        self.endpoint_weight = float(endpoint_weight)
        self.edge_token_weight = float(edge_token_weight)
        self.theta = float(theta)
        self.token_class_emb = nn.Embedding(self.token_class_buckets, dim)
        self.position_bucket_emb = nn.Embedding(self.position_buckets, dim)
        self.indegree_emb = nn.Embedding(max(1, self.edge_radius + 1), dim)
        self.edge_distance_emb = nn.Embedding(max(1, self.edge_radius + 1), dim)
        self.edge_source_class_emb = nn.Embedding(self.token_class_buckets, dim)
        self.torus_proj = CastedLinear(4, dim, bias=False)
        self.identifier_proj = CastedLinear(self.identifier_feature_dim, dim, bias=False)
        self.endpoint_proj = CastedLinear(2 * self.identifier_feature_dim + 1, dim, bias=False)
        self.edge_token_proj = CastedLinear(dim, dim, bias=False)
        self.identifier_proj._zero_init = True
        self.endpoint_proj._zero_init = True
        self.edge_token_proj._zero_init = True

    def _torus_features(self, positions: Tensor, dtype: torch.dtype) -> Tensor:
        pos = positions.to(dtype=torch.float32)
        theta = 2.0 * math.pi * self.theta * pos
        beta = 2.0 * math.pi * 1.4142135623730951 * pos
        features = torch.stack([torch.sin(theta), torch.cos(theta), torch.sin(beta), torch.cos(beta)], dim=-1)
        return features.to(dtype=dtype)

    def _identifier_features(self, positions: Tensor, dtype: torch.dtype) -> Tensor:
        pos = positions.to(dtype=torch.float32)
        freq = torch.arange(1, self.identifier_half_dim + 1, device=positions.device, dtype=torch.float32)
        phase = pos.unsqueeze(-1) * freq.view(*((1,) * pos.ndim), -1) / max(1.0, float(self.position_buckets))
        phase = 2.0 * math.pi * phase
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return (features / math.sqrt(float(self.identifier_feature_dim))).to(dtype=dtype)

    def forward(self, input_ids: Tensor) -> Tensor:
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        token_class = torch.remainder(input_ids.to(torch.long), self.token_class_buckets)
        positions = torch.arange(seqlen, device=device, dtype=torch.long).view(1, seqlen).expand(bsz, seqlen)
        position_bucket = torch.remainder(positions, self.position_buckets)
        indegree = positions.clamp_max(max(0, self.edge_radius))
        structural = (
            self.token_class_emb(token_class)
            + self.position_bucket_emb(position_bucket)
            + self.indegree_emb(indegree)
        )
        identifier_features = self._identifier_features(positions, structural.dtype)
        identifier_context = self.identifier_proj(identifier_features)
        if self.edge_radius > 0:
            edge_context = structural.new_zeros((bsz, seqlen, self.dim))
            endpoint_context = structural.new_zeros((bsz, seqlen, self.dim))
            edge_token_context = structural.new_zeros((bsz, seqlen, self.dim))
            valid_count = structural.new_zeros((bsz, seqlen, 1))
            for distance in range(1, self.edge_radius + 1):
                source_class = token_class.new_zeros((bsz, seqlen))
                source_class[:, distance:] = token_class[:, :-distance]
                valid = (positions >= distance).to(dtype=structural.dtype).unsqueeze(-1)
                dist_ids = torch.full_like(token_class, distance)
                local_edge = (
                    self.edge_source_class_emb(source_class) + self.edge_distance_emb(dist_ids)
                )
                edge_context = edge_context + valid * local_edge
                source_positions = (positions - distance).clamp_min(0)
                source_id = self._identifier_features(source_positions, structural.dtype)
                distance_feature = torch.full(
                    (*positions.shape, 1),
                    float(distance) / float(max(1, self.edge_radius)),
                    device=device,
                    dtype=structural.dtype,
                )
                endpoint_context = endpoint_context + valid * self.endpoint_proj(
                    torch.cat([source_id, identifier_features, distance_feature], dim=-1)
                )
                edge_token_context = edge_token_context + valid * self.edge_token_proj(local_edge)
                valid_count = valid_count + valid
            edge_context = edge_context / valid_count.clamp_min(1.0)
            endpoint_context = endpoint_context / valid_count.clamp_min(1.0)
            edge_token_context = edge_token_context / valid_count.clamp_min(1.0)
        else:
            edge_context = structural.new_zeros((bsz, seqlen, self.dim))
            endpoint_context = structural.new_zeros((bsz, seqlen, self.dim))
            edge_token_context = structural.new_zeros((bsz, seqlen, self.dim))
        torus = self.torus_proj(self._torus_features(positions, structural.dtype))
        return (
            float(self.structural_weight) * structural
            + float(self.edge_weight) * edge_context
            + float(self.torus_weight) * torus
            + float(self.identifier_weight) * identifier_context
            + float(self.endpoint_weight) * endpoint_context
            + float(self.edge_token_weight) * edge_token_context
        )


class GraphOutputFlatteningAdapter(nn.Module):
    """Flatten graph-output states back to sequence order for BPB scoring.

    The transformer hidden table is treated as graph-valued output: each token
    row is a node state and local causal one-dimensional edges aggregate from
    already decoded source nodes.  The adapter returns the same row order used
    by SentencePiece BPB evaluation, so the OAI competition objective remains a
    standard next-token sequence score while the model learns a graph-in,
    graph-out representation internally.
    """

    def __init__(
        self,
        dim: int,
        *,
        edge_radius: int,
        node_weight: float,
        edge_weight: float,
        virtual_edge_tokens: bool,
        edge_token_weight: float,
        score_correction: bool,
        score_correction_weight: float,
    ) -> None:
        super().__init__()
        if edge_radius < 0:
            raise ValueError("GRAPH_OUTPUT_EDGE_RADIUS must be nonnegative")
        self.dim = int(dim)
        self.edge_radius = int(edge_radius)
        self.node_weight = float(node_weight)
        self.edge_weight = float(edge_weight)
        self.virtual_edge_tokens = bool(virtual_edge_tokens)
        self.edge_token_weight = float(edge_token_weight)
        self.score_correction = bool(score_correction)
        self.score_correction_weight = float(score_correction_weight)
        self.node_proj = CastedLinear(dim, dim, bias=False)
        self.edge_proj = CastedLinear(dim, dim, bias=False)
        self.edge_distance_emb = nn.Embedding(max(1, self.edge_radius + 1), dim)
        self.node_proj._zero_init = True
        self.edge_proj._zero_init = True
        self.gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.edge_token_gate = nn.Parameter(torch.tensor(-2.5, dtype=torch.float32))
        self.score_correction_gate = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))

    def forward(self, graph_hidden: Tensor) -> Tensor:
        if self.edge_radius > 0:
            edge_context = graph_hidden.new_zeros(graph_hidden.shape)
            edge_token_context = graph_hidden.new_zeros(graph_hidden.shape)
            valid_count = graph_hidden.new_zeros((*graph_hidden.shape[:2], 1))
            for distance in range(1, self.edge_radius + 1):
                valid = torch.zeros((*graph_hidden.shape[:2], 1), device=graph_hidden.device, dtype=graph_hidden.dtype)
                valid[:, distance:, :] = 1
                shifted = graph_hidden.new_zeros(graph_hidden.shape)
                shifted[:, distance:, :] = graph_hidden[:, :-distance, :]
                edge_context = edge_context + valid * shifted
                if self.virtual_edge_tokens:
                    dist_ids = torch.full(
                        graph_hidden.shape[:2],
                        distance,
                        device=graph_hidden.device,
                        dtype=torch.long,
                    )
                    virtual_edge = (
                        shifted
                        + graph_hidden
                        + self.edge_distance_emb(dist_ids).to(dtype=graph_hidden.dtype)
                    )
                    edge_token_context = edge_token_context + valid * virtual_edge
                valid_count = valid_count + valid
            edge_context = edge_context / valid_count.clamp_min(1.0)
            edge_token_context = edge_token_context / valid_count.clamp_min(1.0)
        else:
            edge_context = graph_hidden.new_zeros(graph_hidden.shape)
            edge_token_context = graph_hidden.new_zeros(graph_hidden.shape)
        update = (
            float(self.node_weight) * self.node_proj(graph_hidden)
            + float(self.edge_weight) * self.edge_proj(edge_context)
        )
        if self.virtual_edge_tokens:
            edge_gate = torch.sigmoid(self.edge_token_gate).to(dtype=graph_hidden.dtype)
            update = update + edge_gate * float(self.edge_token_weight) * self.edge_proj(edge_token_context)
        if self.score_correction:
            score_gate = torch.sigmoid(self.score_correction_gate).to(dtype=graph_hidden.dtype)
            update = update + score_gate * float(self.score_correction_weight) * self.node_proj(graph_hidden - edge_context)
        gate = torch.sigmoid(self.gate).to(dtype=graph_hidden.dtype)
        return graph_hidden + gate * update


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        fineweb_graphify: bool,
        tokengt_first_class: bool,
        tokengt_graph_radius: int,
        tokengt_token_class_buckets: int,
        tokengt_position_buckets: int,
        tokengt_structural_weight: float,
        tokengt_edge_weight: float,
        tokengt_torus_weight: float,
        tokengt_identifier_dim: int,
        tokengt_identifier_weight: float,
        tokengt_endpoint_weight: float,
        tokengt_edge_token_weight: float,
        graph_output_flattening: bool,
        graph_output_edge_radius: int,
        graph_output_node_weight: float,
        graph_output_edge_weight: float,
        graph_output_virtual_edge_tokens: bool,
        graph_output_edge_token_weight: float,
        graph_output_score_correction: bool,
        graph_output_score_correction_weight: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.fineweb_graphify = bool(fineweb_graphify)
        self.tokengt_first_class = bool(tokengt_first_class and fineweb_graphify)
        self.graph_output_flattening_enabled = bool(graph_output_flattening and fineweb_graphify)
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.fineweb_tokengt = (
            FineWebTokenGTStructuralEmbedding(
                model_dim,
                token_class_buckets=tokengt_token_class_buckets,
                position_buckets=tokengt_position_buckets,
                edge_radius=tokengt_graph_radius,
                structural_weight=tokengt_structural_weight,
                edge_weight=tokengt_edge_weight,
                torus_weight=tokengt_torus_weight,
                identifier_dim=tokengt_identifier_dim,
                identifier_weight=tokengt_identifier_weight,
                endpoint_weight=tokengt_endpoint_weight,
                edge_token_weight=tokengt_edge_token_weight,
            )
            if self.tokengt_first_class
            else None
        )
        self.graph_output_flattening = (
            GraphOutputFlatteningAdapter(
                model_dim,
                edge_radius=graph_output_edge_radius,
                node_weight=graph_output_node_weight,
                edge_weight=graph_output_edge_weight,
                virtual_edge_tokens=graph_output_virtual_edge_tokens,
                edge_token_weight=graph_output_edge_token_weight,
                score_correction=graph_output_score_correction,
                score_correction_weight=graph_output_score_correction_weight,
            )
            if self.graph_output_flattening_enabled
            else None
        )
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _hidden(self, input_ids: Tensor, *, flatten_graph_output: bool) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.fineweb_tokengt is not None:
            x = x + self.fineweb_tokengt(input_ids).to(dtype=x.dtype)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x)
        if flatten_graph_output and self.graph_output_flattening is not None:
            x = self.graph_output_flattening(x)
        return x

    def _logits_from_hidden(self, hidden: Tensor) -> Tensor:
        x = hidden.reshape(-1, hidden.size(-1))
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def _loss_from_hidden(self, hidden: Tensor, target_ids: Tensor) -> tuple[Tensor, Tensor]:
        targets = target_ids.reshape(-1)
        logits = self._logits_from_hidden(hidden)
        per_token_nll = F.cross_entropy(logits.float(), targets, reduction="none").view_as(target_ids)
        return per_token_nll.mean(), per_token_nll

    def logits_for_distill(self, input_ids: Tensor, *, flatten_graph_output: bool = True) -> Tensor:
        hidden = self._hidden(input_ids, flatten_graph_output=flatten_graph_output)
        return self._logits_from_hidden(hidden).view(*input_ids.shape, -1)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        hidden = self._hidden(input_ids, flatten_graph_output=True)
        loss, _ = self._loss_from_hidden(hidden, target_ids)
        return loss

    def forward_aux(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        *,
        flatten_graph_output: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self._hidden(input_ids, flatten_graph_output=flatten_graph_output)
        loss, per_token_nll = self._loss_from_hidden(hidden, target_ids)
        return loss, hidden, per_token_nll


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg, flush=True)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    wandb_run = None
    if master_process and os.environ.get("WANDB_PROJECT"):
        try:
            import wandb

            wandb_config = {}
            for name in dir(args):
                if name.startswith("_"):
                    continue
                value = getattr(args, name)
                if isinstance(value, (str, int, float, bool)):
                    wandb_config[name] = value

            wandb_run = wandb.init(
                project=os.environ["WANDB_PROJECT"],
                entity=os.environ.get("WANDB_ENTITY") or None,
                name=os.environ.get("WANDB_RUN_NAME", args.run_id),
                id=os.environ.get("WANDB_RUN_ID", args.run_id),
                resume=os.environ.get("WANDB_RESUME", "allow"),
                config=wandb_config,
            )
        except Exception as exc:
            log0(f"wandb:init_failed:{exc!r}")

    def wandb_log0(payload: dict[str, float], step_value: int) -> None:
        if not master_process or wandb_run is None:
            return
        try:
            wandb_run.log(payload, step=step_value)
        except Exception as exc:
            log0(f"wandb:log_failed:{exc!r}")

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    caseops_token_map = build_caseops_token_map(sp, args.vocab_size) if args.fineweb_caseops else None
    caseops_pairs = 0 if caseops_token_map is None else int((caseops_token_map != torch.arange(args.vocab_size)).sum().item())
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, token_map=caseops_token_map)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(f"fineweb_caseops:enabled:{int(args.fineweb_caseops)} swapped_token_ids:{caseops_pairs}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        fineweb_graphify=args.fineweb_graphify,
        tokengt_first_class=args.tokengt_first_class,
        tokengt_graph_radius=args.tokengt_graph_radius,
        tokengt_token_class_buckets=args.tokengt_token_class_buckets,
        tokengt_position_buckets=args.tokengt_position_buckets,
        tokengt_structural_weight=args.tokengt_structural_weight,
        tokengt_edge_weight=args.tokengt_edge_weight,
        tokengt_torus_weight=args.tokengt_torus_weight,
        tokengt_identifier_dim=args.tokengt_identifier_dim,
        tokengt_identifier_weight=args.tokengt_identifier_weight,
        tokengt_endpoint_weight=args.tokengt_endpoint_weight,
        tokengt_edge_token_weight=args.tokengt_edge_token_weight,
        graph_output_flattening=args.graph_output_flattening,
        graph_output_edge_radius=args.graph_output_edge_radius,
        graph_output_node_weight=args.graph_output_node_weight,
        graph_output_edge_weight=args.graph_output_edge_weight,
        graph_output_virtual_edge_tokens=args.graph_output_virtual_edge_tokens,
        graph_output_edge_token_weight=args.graph_output_edge_token_weight,
        graph_output_score_correction=args.graph_output_score_correction,
        graph_output_score_correction_weight=args.graph_output_score_correction_weight,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    sidecar = None
    graph_loader = None
    graph_lm_loader = None
    if args.graph_lm_primary:
        if args.graph_lm_batch_size < 1:
            raise ValueError("GRAPH_LM_BATCH_SIZE must be at least 1")
        if args.graph_lm_seq_len > args.train_seq_len:
            raise ValueError("GRAPH_LM_SEQ_LEN must not exceed TRAIN_SEQ_LEN")
        try:
            graph_lm_loader = GraphParquetTokenStream(
                args.graph_train_glob,
                sp,
                args.graph_lm_seq_len,
                args.graph_lm_batch_size,
            )
        except Exception:
            if args.require_graph_lm_primary:
                raise
            log0("graph_lm_primary:init_failed_nonfatal enabled=0")
            graph_lm_loader = None
    if args.toricgt_sidecar:
        if args.sidecar_batch_size < 2:
            raise ValueError("TORICGT_SIDECAR_BATCH_SIZE must be at least 2 so the retrieval head has candidates")
        if args.sidecar_seq_len > args.train_seq_len:
            raise ValueError("TORICGT_SIDECAR_SEQ_LEN must not exceed TRAIN_SEQ_LEN")
        try:
            sidecar = ToricGTSidecar(args.model_dim, args).to(device)
            graph_loader = GraphParquetTokenStream(args.graph_train_glob, sp, args.sidecar_seq_len, args.sidecar_batch_size)
        except Exception:
            if args.require_toricgt_sidecar:
                raise
            log0("toricgt_sidecar:init_failed_nonfatal enabled=0")
            sidecar = None
            graph_loader = None
    if args.require_toricgt_sidecar and sidecar is None:
        raise RuntimeError("REQUIRE_TORICGT_SIDECAR=1 but the ToricGT sidecar is disabled")
    if args.require_graph_lm_primary and graph_lm_loader is None:
        raise RuntimeError("REQUIRE_GRAPH_LM_PRIMARY=1 but the graph LM primary stream is disabled")
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            base_model.load_state_dict(checkpoint["model"], strict=True)
            if sidecar is not None and checkpoint.get("sidecar") is not None:
                sidecar.load_state_dict(checkpoint["sidecar"], strict=True)
            if args.start_step <= 0:
                args.start_step = int(checkpoint.get("step", 0))
        else:
            base_model.load_state_dict(checkpoint, strict=True)
        log0(f"resume_checkpoint:{resume_path} start_step:{args.start_step}")
    teacher_model: GPT | None = None
    if args.teacher_checkpoint and args.teacher_distill_weight > 0.0:
        teacher_path = Path(args.teacher_checkpoint)
        teacher_model = GPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            fineweb_graphify=args.fineweb_graphify,
            tokengt_first_class=args.tokengt_first_class,
            tokengt_graph_radius=args.tokengt_graph_radius,
            tokengt_token_class_buckets=args.tokengt_token_class_buckets,
            tokengt_position_buckets=args.tokengt_position_buckets,
            tokengt_structural_weight=args.tokengt_structural_weight,
            tokengt_edge_weight=args.tokengt_edge_weight,
            tokengt_torus_weight=args.tokengt_torus_weight,
            tokengt_identifier_dim=args.tokengt_identifier_dim,
            tokengt_identifier_weight=args.tokengt_identifier_weight,
            tokengt_endpoint_weight=args.tokengt_endpoint_weight,
            tokengt_edge_token_weight=args.tokengt_edge_token_weight,
            graph_output_flattening=args.graph_output_flattening,
            graph_output_edge_radius=args.graph_output_edge_radius,
            graph_output_node_weight=args.graph_output_node_weight,
            graph_output_edge_weight=args.graph_output_edge_weight,
            graph_output_virtual_edge_tokens=args.graph_output_virtual_edge_tokens,
            graph_output_edge_token_weight=args.graph_output_edge_token_weight,
            graph_output_score_correction=args.graph_output_score_correction,
            graph_output_score_correction_weight=args.graph_output_score_correction_weight,
        ).to(device).bfloat16()
        teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
        teacher_state = teacher_payload["model"] if isinstance(teacher_payload, dict) and "model" in teacher_payload else teacher_payload
        teacher_missing, teacher_unexpected = teacher_model.load_state_dict(teacher_state, strict=False)
        for module in teacher_model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        restore_low_dim_params_to_fp32(teacher_model)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        log0(
            f"teacher_distill:enabled checkpoint:{teacher_path} weight:{args.teacher_distill_weight} "
            f"temp:{args.teacher_distill_temperature} missing:{len(teacher_missing)} unexpected:{len(teacher_unexpected)}"
        )
    else:
        log0(
            "teacher_distill:"
            f"enabled:0 checkpoint:{args.teacher_checkpoint or 'none'} weight:{args.teacher_distill_weight}"
        )
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    first_class_tokengt_params = (
        list(base_model.fineweb_tokengt.parameters())
        if getattr(base_model, "fineweb_tokengt", None) is not None
        else []
    )
    graph_output_flattening_params = (
        list(base_model.graph_output_flattening.parameters())
        if getattr(base_model, "graph_output_flattening", None) is not None
        else []
    )
    graph_structure_groups = []
    if first_class_tokengt_params:
        graph_structure_groups.append(
            {
                "params": first_class_tokengt_params,
                "lr": args.tokengt_first_class_lr,
                "base_lr": args.tokengt_first_class_lr,
            }
        )
    if graph_output_flattening_params:
        graph_structure_groups.append(
            {
                "params": graph_output_flattening_params,
                "lr": args.graph_output_flattening_lr,
                "base_lr": args.graph_output_flattening_lr,
            }
        )
    if graph_structure_groups:
        optimizer_first_class_tokengt = torch.optim.AdamW(
            graph_structure_groups,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_first_class_tokengt)
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)
    if sidecar is not None:
        optimizer_sidecar = torch.optim.AdamW(
            [{"params": list(sidecar.parameters()), "lr": args.sidecar_lr, "base_lr": args.sidecar_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_sidecar)

    n_params = sum(p.numel() for p in base_model.parameters())
    first_class_tokengt_param_count = sum(p.numel() for p in first_class_tokengt_params)
    graph_output_flattening_param_count = sum(p.numel() for p in graph_output_flattening_params)
    sidecar_params = sum(p.numel() for p in sidecar.parameters()) if sidecar is not None else 0
    log0(f"model_params:{n_params}")
    log0(
        "fineweb_graphification:"
        f"enabled:{int(args.fineweb_graphify)} tokengt_first_class:{int(args.tokengt_first_class)} "
        f"params:{first_class_tokengt_param_count} radius:{args.tokengt_graph_radius} "
        f"token_class_buckets:{args.tokengt_token_class_buckets} position_buckets:{args.tokengt_position_buckets} "
        f"weights:structural={args.tokengt_structural_weight},edge={args.tokengt_edge_weight},"
        f"torus={args.tokengt_torus_weight},identifier={args.tokengt_identifier_weight},"
        f"endpoint={args.tokengt_endpoint_weight},edge_token={args.tokengt_edge_token_weight} "
        f"identifier_dim:{args.tokengt_identifier_dim} lr:{args.tokengt_first_class_lr}"
    )
    log0(
        "graph_output_flattening:"
        f"enabled:{int(base_model.graph_output_flattening is not None)} "
        f"params:{graph_output_flattening_param_count} radius:{args.graph_output_edge_radius} "
        f"weights:node={args.graph_output_node_weight},edge={args.graph_output_edge_weight},"
        f"virtual_edge={args.graph_output_edge_token_weight},"
        f"score_correction={args.graph_output_score_correction_weight} "
        f"virtual_edge_tokens:{int(args.graph_output_virtual_edge_tokens)} "
        f"score_correction_enabled:{int(args.graph_output_score_correction)} "
        f"lr:{args.graph_output_flattening_lr} flatten_order=sentencepiece_sequence scope=oai_fineweb_bpb_only"
    )
    log0(
        "graph_lm_primary:"
        f"enabled:{int(graph_lm_loader is not None)} required:{int(args.require_graph_lm_primary)} "
        f"seq_len:{args.graph_lm_seq_len} batch_size:{args.graph_lm_batch_size} every:{args.graph_lm_every} "
        f"loss_weight_start:{args.graph_lm_loss_weight_start} loss_weight_peak:{args.graph_lm_loss_weight} "
        f"warmup_steps:{args.graph_lm_warmup_steps} hold_steps:{args.graph_lm_hold_steps} "
        f"graph_glob:{args.graph_train_glob} "
        f"stream:{graph_lm_loader.describe() if graph_lm_loader is not None else 'disabled'}"
    )
    log0(
        "toricgt_sidecar:"
        f"enabled:{int(sidecar is not None)} required:{int(args.require_toricgt_sidecar)} "
        f"params:{sidecar_params} graph_glob:{args.graph_train_glob} "
        f"seq_len:{args.sidecar_seq_len} batch_size:{args.sidecar_batch_size} every:{args.sidecar_every} "
        f"weights:graphcg={args.graphcg_loss_weight},analogy={args.analogy_loss_weight},"
        f"tokengt_graph={args.tokengt_graph_loss_weight},trajectory_memory={args.trajectory_memory_loss_weight},"
        f"toric_geometry={args.toric_geometry_loss_weight},vector_bundle_1d_cone_ce={args.toric_vector_bundle_loss_weight},"
        f"toric_bgg={args.toric_bgg_loss_weight},koszul={args.koszul_persistence_loss_weight},"
        f"combinatorial_toric={args.combinatorial_toric_loss_weight},"
        f"compute_all_metrics={args.sidecar_compute_all_metrics}"
    )
    log0(
        "advanced_bpb_controls:"
        f"aux_grad_routing:{int(args.aux_grad_routing)} graph_lm:{int(args.aux_grad_route_graph_lm)} "
        f"sidecar:{int(args.aux_grad_route_sidecar)} teacher:{int(args.aux_grad_route_teacher)} "
        f"retrieval_conditioned_aux:{int(args.retrieval_conditioned_aux)} "
        f"uncertainty_weighting:{int(args.sidecar_uncertainty_weighting)} "
        f"graphcg_bpb_orthogonal_weight:{args.graphcg_bpb_orthogonal_weight} "
        f"score_first_tta:{int(args.score_first_tta)}"
    )
    log0(
        "toricgt_sidecar_heads:"
        f"graphcg_full_rank:{int(sidecar.graphcg_basis.shape[0]) if sidecar is not None else 0} "
        f"analogy_lattice:{int(sidecar is not None)} "
        f"tokengt_causal_graph:{int(sidecar is not None)} "
        f"trajectory_memory_retrieval:{int(sidecar is not None)} "
        f"toric_geometry:{int(sidecar is not None)} "
        f"toric_vector_bundle_1d_cone_sheaf:{int(sidecar is not None)} "
        f"toric_bgg_category_o:{int(sidecar is not None)} "
        f"koszul_persistence:{int(sidecar is not None)} "
        f"combinatorial_toric_commutative_algebra:{int(sidecar is not None)}"
    )
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f} start_step:{args.start_step}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device, token_map=caseops_token_map)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    base_grad_params = [param for param in base_model.parameters() if param.requires_grad]

    def snapshot_base_grads() -> list[Tensor | None]:
        return [None if param.grad is None else param.grad.detach().clone() for param in base_grad_params]

    def zero_base_grads() -> None:
        for param in base_grad_params:
            param.grad = None

    def restore_base_grads(grads: list[Tensor | None]) -> None:
        for param, grad in zip(base_grad_params, grads, strict=True):
            param.grad = None if grad is None else grad.clone()

    def add_base_grads(grads: list[Tensor | None]) -> None:
        for param, grad in zip(base_grad_params, grads, strict=True):
            if grad is None:
                continue
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.add_(grad.to(dtype=param.grad.dtype))

    def route_aux_grads(
        primary: list[Tensor | None] | None,
        aux: list[Tensor | None],
        *,
        name: str,
    ) -> tuple[list[Tensor | None], dict[str, float]]:
        if primary is None or not args.aux_grad_routing:
            aux_norm = math.sqrt(
                sum(float(grad.float().pow(2).sum().item()) for grad in aux if grad is not None)
            )
            return aux, {
                f"aux_grad_routing/{name}_enabled": 0.0,
                f"aux_grad_routing/{name}_cosine": 0.0,
                f"aux_grad_routing/{name}_conflict": 0.0,
                f"aux_grad_routing/{name}_projection_coeff": 0.0,
                f"aux_grad_routing/{name}_routed_norm": aux_norm,
            }
        metric_device = next((grad.device for grad in aux if grad is not None), device)
        dot_t = torch.zeros((), device=metric_device, dtype=torch.float32)
        primary_norm_sq_t = torch.zeros((), device=metric_device, dtype=torch.float32)
        aux_norm_sq_t = torch.zeros((), device=metric_device, dtype=torch.float32)
        for p_grad, a_grad in zip(primary, aux, strict=True):
            if p_grad is None or a_grad is None:
                continue
            pf = p_grad.float()
            af = a_grad.float()
            dot_t = dot_t + (pf * af).sum()
            primary_norm_sq_t = primary_norm_sq_t + pf.pow(2).sum()
            aux_norm_sq_t = aux_norm_sq_t + af.pow(2).sum()
        dot = float(dot_t.detach().item())
        primary_norm_sq = float(primary_norm_sq_t.detach().item())
        aux_norm_sq = float(aux_norm_sq_t.detach().item())
        cosine = dot / (math.sqrt(max(primary_norm_sq, 1e-30)) * math.sqrt(max(aux_norm_sq, 1e-30)))
        conflict = dot < 0.0
        coeff = dot / max(primary_norm_sq, 1e-30) if conflict and args.aux_grad_conflict_projection else 0.0
        routed: list[Tensor | None] = []
        routed_norm_sq_t = torch.zeros((), device=metric_device, dtype=torch.float32)
        for p_grad, a_grad in zip(primary, aux, strict=True):
            if a_grad is None:
                routed.append(None)
                continue
            if conflict and args.aux_grad_conflict_projection and p_grad is not None:
                r = (a_grad.float() - coeff * p_grad.float()).to(dtype=a_grad.dtype)
            else:
                r = (float(args.aux_grad_aligned_boost) * a_grad.float()).to(dtype=a_grad.dtype)
            routed_norm_sq_t = routed_norm_sq_t + r.float().pow(2).sum()
            routed.append(r)
        routed_norm_sq = float(routed_norm_sq_t.detach().item())
        return routed, {
            f"aux_grad_routing/{name}_enabled": 1.0,
            f"aux_grad_routing/{name}_cosine": float(cosine),
            f"aux_grad_routing/{name}_conflict": float(conflict),
            f"aux_grad_routing/{name}_projection_coeff": float(coeff),
            f"aux_grad_routing/{name}_primary_norm": math.sqrt(max(primary_norm_sq, 0.0)),
            f"aux_grad_routing/{name}_aux_norm": math.sqrt(max(aux_norm_sq, 0.0)),
            f"aux_grad_routing/{name}_routed_norm": math.sqrt(max(routed_norm_sq, 0.0)),
        }

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def linear_schedule(step_value: int, start: float, peak: float, warmup_steps: int, hold_steps: int = 0) -> float:
        if warmup_steps <= 0:
            return peak
        if step_value < hold_steps:
            return start
        frac = min(max((step_value - hold_steps) / max(warmup_steps, 1), 0.0), 1.0)
        return start + frac * (peak - start)

    def teacher_weight(step_value: int) -> float:
        if teacher_model is None or args.teacher_distill_weight <= 0.0:
            return 0.0
        if args.teacher_distill_until_step > 0 and step_value >= args.teacher_distill_until_step:
            return 0.0
        peak = float(args.teacher_distill_weight)
        start = float(args.teacher_distill_weight_start)
        if args.teacher_distill_decay_steps <= 0:
            return peak
        frac = min(max(step_value / max(args.teacher_distill_decay_steps, 1), 0.0), 1.0)
        return start + frac * (peak - start)

    def save_checkpoint(step_value: int, val_loss_value: float | None = None, val_bpb_value: float | None = None) -> None:
        if not master_process or args.checkpoint_every <= 0:
            return
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step_value),
            "model": base_model.state_dict(),
            "sidecar": sidecar.state_dict() if sidecar is not None else None,
            "toricgt_sidecar_enabled": bool(sidecar is not None),
            "graph_lm_primary_enabled": bool(graph_lm_loader is not None),
            "graph_output_flattening_enabled": bool(base_model.graph_output_flattening is not None),
            "val_loss": val_loss_value,
            "val_bpb": val_bpb_value,
            "run_id": args.run_id,
            "hyperparameters": {
                key: value for key, value in vars(args).items() if isinstance(value, (str, int, float, bool))
            },
        }
        path = ckpt_dir / f"{args.run_id}_step_{int(step_value):06d}.pt"
        torch.save(payload, path)
        log0(f"checkpoint_saved:{path} step:{step_value} val_bpb:{val_bpb_value}")

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0 and args.start_step <= 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device, token_map=caseops_token_map)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = int(args.start_step)
    while True:
        last_step = step >= args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms "
                f"step_avg:{training_time_ms / max(step - args.start_step, 1):.2f}ms"
            )
            wandb_log0(
                {
                    "val/loss": float(val_loss),
                    "val/bpb": float(val_bpb),
                    "oai_competition/bpb": float(val_bpb),
                    "bpb/oai_competition": float(val_bpb),
                    "00_primary/val_bpb": float(val_bpb),
                },
                step,
            )
            save_checkpoint(step, val_loss, val_bpb)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        train_token_count = torch.zeros((), device=device, dtype=torch.float64)
        train_byte_count = torch.zeros((), device=device, dtype=torch.float64)
        graph_lm_metrics: dict[str, float] = {}
        sidecar_metrics: dict[str, float] = {}
        teacher_metrics: dict[str, float] = {}
        aux_grad_metrics: dict[str, float] = {}
        last_fineweb_x: Tensor | None = None
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            last_fineweb_x = x
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            batch_token_count = float(y.numel())
            train_loss_sum += loss.detach().to(torch.float64) * batch_token_count
            train_token_count += batch_token_count
            train_byte_count += count_sentencepiece_target_bytes(
                x,
                y,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            (loss * grad_scale).backward()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_token_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_byte_count, op=dist.ReduceOp.SUM)
        train_loss = (train_loss_sum / train_token_count.clamp_min(1.0)).to(dtype=torch.float32)
        train_bits_per_token = float(train_loss.detach().float().item() / math.log(2.0))
        train_tokens_per_byte = float((train_token_count / train_byte_count.clamp_min(1.0)).detach().float().item())
        train_bpb = float(train_bits_per_token * train_tokens_per_byte)
        primary_grads = snapshot_base_grads() if args.aux_grad_routing else None

        current_teacher_weight = teacher_weight(step)
        if teacher_model is not None and current_teacher_weight > 0.0 and last_fineweb_x is not None:
            teacher_model.eval()
            seq_count = min(max(1, int(args.teacher_distill_max_sequences)), last_fineweb_x.shape[0])
            dx = last_fineweb_x[:seq_count]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                student_logits = base_model.logits_for_distill(dx, flatten_graph_output=True)
                with torch.no_grad():
                    teacher_logits = teacher_model.logits_for_distill(dx, flatten_graph_output=True)
                temp = max(float(args.teacher_distill_temperature), 1e-4)
                teacher_flat = teacher_logits.float().reshape(-1, teacher_logits.shape[-1])
                student_flat = student_logits.float().reshape(-1, student_logits.shape[-1])
                teacher_probs = torch.softmax(teacher_flat / temp, dim=-1)
                student_log_probs = torch.log_softmax(student_flat / temp, dim=-1)
                teacher_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)
                weighted_teacher_loss = float(current_teacher_weight) * teacher_kl
            if args.aux_grad_routing and args.aux_grad_route_teacher:
                saved_base_grads = snapshot_base_grads()
                zero_base_grads()
                weighted_teacher_loss.backward()
                aux = snapshot_base_grads()
                routed, metrics = route_aux_grads(primary_grads, aux, name="teacher")
                restore_base_grads(saved_base_grads)
                add_base_grads(routed)
                aux_grad_metrics.update(metrics)
            else:
                weighted_teacher_loss.backward()
            teacher_metrics = {
                "teacher_distill/enabled": 1.0,
                "teacher_distill/kl": float(teacher_kl.detach().float().item()),
                "teacher_distill/weighted_loss": float(weighted_teacher_loss.detach().float().item()),
                "teacher_distill/weight": float(current_teacher_weight),
                "teacher_distill/temperature": float(args.teacher_distill_temperature),
                "teacher_distill/sequences": float(seq_count),
            }

        scheduled_graph_lm_weight = linear_schedule(
            step,
            args.graph_lm_loss_weight_start,
            args.graph_lm_loss_weight,
            args.graph_lm_warmup_steps,
            args.graph_lm_hold_steps,
        )
        if (
            graph_lm_loader is not None
            and args.graph_lm_every > 0
            and step % args.graph_lm_every == 0
            and scheduled_graph_lm_weight > 0
        ):
            gx, gy = graph_lm_loader.next_batch(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                graph_lm_loss, _, _ = base_model.forward_aux(gx, gy)
                weighted_graph_lm_loss = float(scheduled_graph_lm_weight) * graph_lm_loss
            graph_byte_count = count_sentencepiece_target_bytes(
                gx,
                gy,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            graph_bits_per_token = float(graph_lm_loss.detach().float().item() / math.log(2.0))
            graph_tokens_per_byte = float(gy.numel() / graph_byte_count.detach().float().clamp_min(1.0).item())
            graph_lm_bpb = float(graph_bits_per_token * graph_tokens_per_byte)
            if args.aux_grad_routing and args.aux_grad_route_graph_lm:
                saved_base_grads = snapshot_base_grads()
                zero_base_grads()
                weighted_graph_lm_loss.backward()
                aux = snapshot_base_grads()
                routed, metrics = route_aux_grads(primary_grads, aux, name="graph_lm")
                restore_base_grads(saved_base_grads)
                add_base_grads(routed)
                aux_grad_metrics.update(metrics)
            else:
                weighted_graph_lm_loss.backward()
            graph_lm_metrics = {
                "graph_lm_primary/loss": float(graph_lm_loss.detach().float().item()),
                "graph_lm_primary/weighted_loss": float(weighted_graph_lm_loss.detach().float().item()),
                "graph_lm_primary/bpb": float(graph_lm_bpb),
                "graph_lm_primary/bits_per_token": float(graph_bits_per_token),
                "graph_lm_primary/tokens_per_byte": float(graph_tokens_per_byte),
                "graph_lm_primary/target_tokens": float(gy.numel()),
                "graph_lm_primary/target_bytes": float(graph_byte_count.detach().float().item()),
                "graph_lm_primary/loss_weight": float(scheduled_graph_lm_weight),
                "graph_lm_primary/loss_weight_peak": float(args.graph_lm_loss_weight),
                "graph_lm_primary/loss_weight_start": float(args.graph_lm_loss_weight_start),
                "graph_lm_primary/warmup_steps": float(args.graph_lm_warmup_steps),
                "graph_lm_primary/seq_len": float(args.graph_lm_seq_len),
                "graph_lm_primary/batch_size": float(args.graph_lm_batch_size),
            }

        if sidecar is not None and graph_loader is not None and args.sidecar_every > 0 and step % args.sidecar_every == 0:
            sidecar.train()
            sx, sy = graph_loader.next_batch(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                sidecar_lm_loss, sidecar_hidden, sidecar_nll = base_model.forward_aux(sx, sy)
                positions = torch.arange(sx.shape[1], device=device, dtype=torch.float32).unsqueeze(0).expand(sx.shape[0], -1)
                sidecar_out = sidecar(sidecar_hidden, sy, positions, sidecar_nll)
                sidecar_loss = float(args.sidecar_loss_weight) * sidecar_out["toricgt_sidecar_loss"]
            if args.aux_grad_routing and args.aux_grad_route_sidecar:
                saved_base_grads = snapshot_base_grads()
                zero_base_grads()
                sidecar_loss.backward()
                aux = snapshot_base_grads()
                routed, metrics = route_aux_grads(primary_grads, aux, name="sidecar")
                restore_base_grads(saved_base_grads)
                add_base_grads(routed)
                aux_grad_metrics.update(metrics)
            else:
                sidecar_loss.backward()
            sidecar_metrics = {
                "toricgt_sidecar/lm_loss": float(sidecar_lm_loss.detach().float().item()),
                "toricgt_sidecar/loss": float(sidecar_loss.detach().float().item()),
            }
            for name, value in sidecar_out.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    sidecar_metrics[f"toricgt_sidecar/{name}"] = float(value.detach().float().item())
            if args.require_toricgt_sidecar:
                required = (
                    "toricgt_sidecar/graphcg_loss",
                    "toricgt_sidecar/analogy_lattice_loss",
                    "toricgt_sidecar/tokengt_graph_loss",
                    "toricgt_sidecar/trajectory_memory_loss",
                    "toricgt_sidecar/toric_geometry_loss",
                    "toricgt_sidecar/toric_vector_bundle_1d_cone_ce_loss",
                    "toricgt_sidecar/toric_bgg_loss",
                    "toricgt_sidecar/koszul_persistence_loss",
                    "toricgt_sidecar/toric_cca_topology_loss",
                )
                missing = [name for name in required if name not in sidecar_metrics]
                if missing:
                    raise RuntimeError(f"ToricGT sidecar missing required active metrics: {missing}")

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            graph_lm_brief = ""
            if graph_lm_metrics:
                graph_lm_brief = (
                    f" graph_lm_loss:{graph_lm_metrics.get('graph_lm_primary/loss', 0.0):.6f}"
                    f" graph_lm_bpb:{graph_lm_metrics.get('graph_lm_primary/bpb', 0.0):.4f}"
                    f" graph_lm_w:{graph_lm_metrics.get('graph_lm_primary/loss_weight', 0.0):.4f}"
                )
            teacher_brief = ""
            if teacher_metrics:
                teacher_brief = (
                    f" teacher_kl:{teacher_metrics.get('teacher_distill/kl', 0.0):.6f}"
                    f" teacher_w:{teacher_metrics.get('teacher_distill/weight', 0.0):.4f}"
                )
            sidecar_brief = ""
            if sidecar_metrics:
                sidecar_brief = (
                    f" sidecar_loss:{sidecar_metrics.get('toricgt_sidecar/loss', 0.0):.6f}"
                    f" graphcg:{sidecar_metrics.get('toricgt_sidecar/graphcg_loss', 0.0):.6f}"
                    f" analogy:{sidecar_metrics.get('toricgt_sidecar/analogy_lattice_loss', 0.0):.6f}"
                    f" tokengt_graph:{sidecar_metrics.get('toricgt_sidecar/tokengt_graph_loss', 0.0):.6f}"
                    f" memory:{sidecar_metrics.get('toricgt_sidecar/trajectory_memory_loss', 0.0):.6f}"
                    f" toric:{sidecar_metrics.get('toricgt_sidecar/toric_geometry_loss', 0.0):.6f}"
                    f" vb1d:{sidecar_metrics.get('toricgt_sidecar/toric_vector_bundle_1d_cone_ce_loss', 0.0):.6f}"
                    f" bgg:{sidecar_metrics.get('toricgt_sidecar/toric_bgg_loss', 0.0):.6f}"
                    f" koszul:{sidecar_metrics.get('toricgt_sidecar/koszul_persistence_loss', 0.0):.6f}"
                    f" cca:{sidecar_metrics.get('toricgt_sidecar/toric_cca_topology_loss', 0.0):.6f}"
                )
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_bpb:{train_bpb:.4f} train_bpt:{train_bits_per_token:.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms "
                f"step_avg:{approx_training_time_ms / max(step - args.start_step, 1):.2f}ms"
                f"{graph_lm_brief}"
                f"{teacher_brief}"
                f"{sidecar_brief}"
            )
            payload = {
                "train/loss": float(train_loss.item()),
                "train/bpb": float(train_bpb),
                "train/bits_per_token": float(train_bits_per_token),
                "train/tokens_per_byte": float(train_tokens_per_byte),
                "train/target_tokens": float(train_token_count.detach().float().item()),
                "train/target_bytes": float(train_byte_count.detach().float().item()),
                "train/step_avg_ms": float(approx_training_time_ms / max(step - args.start_step, 1)),
                "fineweb_graphify/enabled": float(args.fineweb_graphify),
                "fineweb_graphify/tokengt_first_class": float(args.tokengt_first_class),
                "fineweb_graphify/graph_radius": float(args.tokengt_graph_radius),
                "fineweb_graphify/token_class_buckets": float(args.tokengt_token_class_buckets),
                "fineweb_graphify/position_buckets": float(args.tokengt_position_buckets),
                "fineweb_graphify/structural_weight": float(args.tokengt_structural_weight),
                "fineweb_graphify/edge_weight": float(args.tokengt_edge_weight),
                "fineweb_graphify/torus_weight": float(args.tokengt_torus_weight),
                "fineweb_graphify/identifier_dim": float(args.tokengt_identifier_dim),
                "fineweb_graphify/identifier_weight": float(args.tokengt_identifier_weight),
                "fineweb_graphify/endpoint_weight": float(args.tokengt_endpoint_weight),
                "fineweb_graphify/edge_token_weight": float(args.tokengt_edge_token_weight),
                "fineweb_graphify/first_class_lr": float(args.tokengt_first_class_lr),
                "graph_output_flattening/enabled": float(base_model.graph_output_flattening is not None),
                "graph_output_flattening/edge_radius": float(args.graph_output_edge_radius),
                "graph_output_flattening/node_weight": float(args.graph_output_node_weight),
                "graph_output_flattening/edge_weight": float(args.graph_output_edge_weight),
                "graph_output_flattening/virtual_edge_tokens": float(args.graph_output_virtual_edge_tokens),
                "graph_output_flattening/edge_token_weight": float(args.graph_output_edge_token_weight),
                "graph_output_flattening/score_correction": float(args.graph_output_score_correction),
                "graph_output_flattening/score_correction_weight": float(args.graph_output_score_correction_weight),
                "graph_output_flattening/lr": float(args.graph_output_flattening_lr),
                "graph_output_flattening/oai_fineweb_bpb_only": 1.0,
                "graph_lm_primary/enabled": float(graph_lm_loader is not None),
                "graph_lm_primary/required": float(args.require_graph_lm_primary),
                "graph_lm_primary/every": float(args.graph_lm_every),
                "graph_lm_primary/loss_weight_peak_config": float(args.graph_lm_loss_weight),
                "graph_lm_primary/loss_weight_start_config": float(args.graph_lm_loss_weight_start),
                "toricgt_sidecar/enabled": float(sidecar is not None),
                "toricgt_sidecar/graphcg_active": float(sidecar is not None),
                "toricgt_sidecar/analogy_active": float(sidecar is not None),
                "toricgt_sidecar/memory_retrieval_active": float(sidecar is not None),
                "toricgt_sidecar/toric_geometry_weight": float(args.toric_geometry_loss_weight),
                "toricgt_sidecar/toric_vector_bundle_1d_cone_ce_weight": float(args.toric_vector_bundle_loss_weight),
                "toricgt_sidecar/toric_bgg_weight": float(args.toric_bgg_loss_weight),
                "toricgt_sidecar/koszul_persistence_weight": float(args.koszul_persistence_loss_weight),
                "toricgt_sidecar/combinatorial_toric_weight": float(args.combinatorial_toric_loss_weight),
                "teacher_distill/configured": float(args.teacher_distill_weight > 0.0 and bool(args.teacher_checkpoint)),
                "teacher_distill/target_weight": float(args.teacher_distill_weight),
                "fineweb_caseops/enabled": float(args.fineweb_caseops),
                "fineweb_caseops/swapped_token_ids": float(caseops_pairs),
                "aux_grad_routing/enabled": float(args.aux_grad_routing),
                "score_first_tta/enabled": float(args.score_first_tta),
            }
            payload.update(graph_lm_metrics)
            payload.update(teacher_metrics)
            payload.update(sidecar_metrics)
            payload.update(aux_grad_metrics)
            wandb_log0(payload, step)

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
