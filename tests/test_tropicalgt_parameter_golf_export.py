import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


class TropicalGTParameterGolfExportTest(unittest.TestCase):
    def test_smoke_export_is_stripped_and_within_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "export_tropicalgt_parameter_golf.py"),
                "--smoke-dummy",
                "--output-dir",
                tmp,
            ]
            subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
            manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["within_cap"])
            self.assertEqual(
                set(manifest["included_files"]),
                {"train_gpt.py", "tropicalgt_tokengt_adapter.py", "final_model.int8.ptz", "manifest.json"},
            )
            self.assertIn("(token_features, token_type_ids, endpoint_ids, mask)", manifest["graph_conditioning_contract"]["tuple_forms"])
            self.assertEqual(manifest["metric_contract"]["graph_primary"].split()[0], "graph_bpb")
            with zipfile.ZipFile(Path(tmp) / "tropicalgt_parameter_golf_submission.zip") as zf:
                self.assertEqual(set(zf.namelist()), set(manifest["included_files"]))

    def test_export_accepts_nested_tropicalgt_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            torch.save(
                {
                    "model": {
                        "byte_emb.weight": torch.randn(16, 8),
                        "graph_adapter.feature_proj.0.weight": torch.randn(8, 48),
                    },
                    "optimizer": {"state": {}, "param_groups": []},
                    "step": 3,
                },
                checkpoint_path,
            )
            output_dir = Path(tmp) / "export"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "export_tropicalgt_parameter_golf.py"),
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
            ]
            subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["within_cap"])
            self.assertGreater(manifest["artifact_bytes"], 0)
            with zipfile.ZipFile(output_dir / "tropicalgt_parameter_golf_submission.zip") as zf:
                self.assertEqual(set(zf.namelist()), set(manifest["included_files"]))


if __name__ == "__main__":
    unittest.main()
