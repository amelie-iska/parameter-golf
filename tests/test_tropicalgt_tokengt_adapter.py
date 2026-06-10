import math
import unittest

import torch

from tropicalgt_tokengt_adapter import TokenGTAdapterConfig, TokenGTGraphAdapter, graph_bpb_metrics, graph_token_structural_bytes
from train_gpt import GPT


class TropicalGTTokenGTAdapterTest(unittest.TestCase):
    def test_adapter_accepts_endpoint_ids_and_backpropagates(self) -> None:
        adapter = TokenGTGraphAdapter(TokenGTAdapterConfig(graph_feature_dim=6, model_dim=8, max_endpoint_id=16))
        features = torch.randn(2, 4, 6)
        type_ids = torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]])
        mask = torch.ones(2, 4, dtype=torch.bool)
        endpoints = torch.tensor([[[0, -1], [0, 1], [2, -1], [1, 2]], [[0, -1], [0, 2], [1, 2], [3, -1]]])
        out = adapter(features, type_ids, mask, endpoint_ids=endpoints)
        self.assertEqual(tuple(out.shape), (2, 8))
        loss = out.square().mean()
        loss.backward()
        self.assertTrue(any(param.grad is not None for param in adapter.parameters()))

    def test_gpt_accepts_three_and_four_tensor_graph_tuples(self) -> None:
        model = GPT(
            vocab_size=32,
            num_layers=2,
            model_dim=16,
            num_heads=2,
            num_kv_heads=1,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.01,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.0,
            use_graph_adapter=True,
            graph_feature_dim=6,
        )
        x = torch.randint(0, 32, (2, 8))
        y = torch.randint(0, 32, (2, 8))
        features = torch.randn(2, 3, 6)
        type_ids = torch.tensor([[0, 1, 0], [0, 1, 1]])
        endpoints = torch.tensor([[[0, -1], [0, 1], [2, -1]], [[0, -1], [0, 2], [1, 2]]])
        mask = torch.ones(2, 3, dtype=torch.bool)
        loss_three = model(x, y, graph_tokens=(features, type_ids, mask))
        loss_four = model(x, y, graph_tokens=(features, type_ids, endpoints, mask))
        self.assertTrue(torch.isfinite(loss_three))
        self.assertTrue(torch.isfinite(loss_four))

    def test_graph_bpb_metrics_match_formula(self) -> None:
        mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
        type_ids = torch.tensor([[0, 1, 1]])
        endpoints = torch.tensor([[[0, -1], [0, 1], [0, 2]]])
        graph_bytes = graph_token_structural_bytes(mask, type_ids, endpoint_ids=endpoints)
        metrics = graph_bpb_metrics(
            nll=2.0,
            target_bytes=10,
            mask=mask,
            token_type_ids=type_ids,
            endpoint_ids=endpoints,
            explicit_graph_json_bytes=5,
            graph_side_weight=0.5,
        )
        nll_bits = 2.0 * 10 / math.log(2.0)
        self.assertEqual(metrics["graph_token_structural_bytes"], float(graph_bytes))
        self.assertAlmostEqual(metrics["bpb"], nll_bits / 10)
        self.assertAlmostEqual(metrics["graph_bpb"], (nll_bits + 20.0) / (10 + graph_bytes))


if __name__ == "__main__":
    unittest.main()
