#!/usr/bin/env python3
"""Build a stripped TropicalGT Parameter-Golf export.

The research repo keeps GFlowNet, GraphCG, topology, visualization, W&B, and
memory tooling.  The competition export intentionally contains only the files
needed by the OpenAI Parameter-Golf baseline path:

* train_gpt.py
* tropicalgt_tokengt_adapter.py
* a compressed int8+zlib model artifact
* manifest.json with cap and BPB/graph-BPB accounting metadata
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Mapping
from pathlib import Path
import sys
import zipfile
import zlib

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_gpt import quantize_state_dict_int8  # noqa: E402

CAP_BYTES = 16_000_000
CODE_FILES = ("train_gpt.py", "tropicalgt_tokengt_adapter.py")
COMPETITION_EXCLUDED_PREFIXES = ("gfn.", "graphcg.", "memory.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export stripped TropicalGT Parameter-Golf package")
    parser.add_argument("--checkpoint", type=Path, help="PyTorch checkpoint/state_dict to quantize.")
    parser.add_argument("--model-artifact", type=Path, help="Existing final_model.int8.ptz-style artifact to include.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "parameter_golf_export")
    parser.add_argument("--archive-name", default="tropicalgt_parameter_golf_submission.zip")
    parser.add_argument("--cap-bytes", type=int, default=CAP_BYTES)
    parser.add_argument("--smoke-dummy", action="store_true", help="Create a tiny dummy artifact for export tests only.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "final_model.int8.ptz"
    if args.model_artifact:
        artifact_path.write_bytes(args.model_artifact.read_bytes())
    elif args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        artifact_path.write_bytes(_quantized_blob(_extract_tensor_state_dict(state)))
    elif args.smoke_dummy:
        artifact_path.write_bytes(_quantized_blob(_dummy_state_dict()))
    else:
        raise SystemExit("Provide --checkpoint, --model-artifact, or --smoke-dummy.")

    code_sizes = {name: (ROOT / name).stat().st_size for name in CODE_FILES}
    artifact_bytes = artifact_path.stat().st_size
    code_bytes = sum(code_sizes.values())
    manifest = {
        "name": "TropicalGT Parameter-Golf stripped export",
        "cap_bytes": int(args.cap_bytes),
        "artifact_bytes": int(artifact_bytes),
        "code_bytes": int(code_bytes),
        "total_competition_bytes": int(artifact_bytes + code_bytes),
        "within_cap": bool(artifact_bytes + code_bytes <= args.cap_bytes),
        "included_files": [*CODE_FILES, artifact_path.name, "manifest.json"],
        "state_dict_filter": {
            "kept_for_competition": "LM core, graph-token adapter, tropical attention, GRU transition, and output head",
            "excluded_training_only_prefixes": list(COMPETITION_EXCLUDED_PREFIXES),
            "reason": "GFlowNet, GraphCG direction bank, and analogical memory heads are training/audit modules; stripping them preserves BPB inference while keeping the research checkpoint full.",
        },
        "excluded_by_design": [
            "TropicalGT-I research training package",
            "topological algebra audit artifacts",
            "Plotly visualizations",
            "W&B run directories",
            "datasets",
            "checkpoints other than final_model.int8.ptz",
        ],
        "graph_conditioning_contract": {
            "enabled_by": "TROPICALGT_GRAPH_ADAPTER=1",
            "tuple_forms": [
                "(token_features, token_type_ids, mask)",
                "(token_features, token_type_ids, endpoint_ids, mask)",
            ],
            "endpoint_padding": "-1 endpoint ids are padding; nonnegative ids encode TokenGT incidence.",
        },
        "metric_contract": {
            "primary": "val_bpb from train_gpt.py eval_val",
            "graph_primary": "graph_bpb from tropicalgt_tokengt_adapter.graph_bpb_metrics when graph conditioning is evaluated",
            "ablation_rule": "treat auxiliary metrics as predictors; optimize and report BPB/graph-BPB first.",
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive_path = args.output_dir / args.archive_name
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in CODE_FILES:
            zf.write(ROOT / name, arcname=name)
        zf.write(artifact_path, arcname=artifact_path.name)
        zf.write(manifest_path, arcname="manifest.json")
    manifest["archive_bytes"] = archive_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(archive_path), **manifest}, indent=2))
    if not manifest["within_cap"]:
        raise SystemExit(f"Export exceeds cap by {manifest['total_competition_bytes'] - args.cap_bytes} bytes")


def _quantized_blob(state_dict: dict[str, torch.Tensor]) -> bytes:
    competition_state, _stripped = _competition_state_dict(state_dict)
    quant_obj, _stats = quantize_state_dict_int8(competition_state)
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    return zlib.compress(buf.getvalue(), level=9)


def _competition_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    kept: dict[str, torch.Tensor] = {}
    stripped: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith(COMPETITION_EXCLUDED_PREFIXES):
            stripped[key] = tensor
        else:
            kept[key] = tensor
    return kept, stripped


def _extract_tensor_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if _is_tensor_state_dict(checkpoint):
        return dict(checkpoint)
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model", "module"):
            if key in checkpoint:
                nested = checkpoint[key]
                if _is_tensor_state_dict(nested):
                    return dict(nested)
        tensor_like = {str(k): v for k, v in checkpoint.items() if torch.is_tensor(v)}
        if tensor_like and len(tensor_like) == len(checkpoint):
            return tensor_like
    raise TypeError(
        "checkpoint must be a tensor state_dict or contain one under "
        "'model_state_dict', 'state_dict', 'model', or 'module'"
    )


def _is_tensor_state_dict(value: object) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(torch.is_tensor(tensor) for tensor in value.values())


def _dummy_state_dict() -> dict[str, torch.Tensor]:
    return {
        "tok_emb.weight": torch.randn(16, 8),
        "blocks.0.attn.qkv_w": torch.randn(24, 8),
        "blocks.0.attn.c_proj.weight": torch.randn(8, 8),
        "graph_adapter.feature_proj.0.weight": torch.randn(8, 48),
        "graph_adapter.type_emb.weight": torch.randn(4, 8),
        "graph_adapter.endpoint_emb.weight": torch.randn(16, 8),
    }


if __name__ == "__main__":
    main()
