# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
from executorch.examples.models.llama.export_llama_lib import (
    _prepare_for_llama_export,
)
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model import Llama2Model
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.extension.llm.export.config.llm_config import (
    DtypeOverride,
    LlmConfig,
    ModelType,
)


class CheckpointQuantizationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.params_path = self.root / "params.json"
        params = {
            "dim": 16,
            "hidden_dim": 32,
            "multiple_of": 4,
            "n_heads": 4,
            "n_layers": 1,
            "norm_eps": 1e-5,
            "vocab_size": 32,
        }
        self.params_path.write_text(json.dumps(params))

        model_args = ModelArgs(
            max_seq_len=8,
            max_context_len=8,
            max_batch_size=1,
            use_kv_cache=False,
            use_sdpa_with_kv_cache_op=False,
            generate_full_logits=False,
            enable_dynamic_shape=False,
            **params,
        )
        checkpoint_path = self.root / "model.pth"
        torch.save(construct_transformer(model_args).state_dict(), checkpoint_path)

        issue_dir = self.root / "Qwen3-0.6B-int8-int4-unsloth-torchao"
        issue_dir.mkdir()
        self.checkpoint_paths = [
            checkpoint_path,
            self.root / "model-int8.pth",
            self.root / "model-8da4w.pth",
            issue_dir / "pytorch_model_converted.bin",
        ]
        for path in self.checkpoint_paths[1:]:
            shutil.copyfile(checkpoint_path, path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _config(self, checkpoint_path: Path, qmode=None) -> LlmConfig:
        config = LlmConfig()
        config.base.model_class = ModelType.llama2
        config.base.checkpoint = str(checkpoint_path)
        config.base.params = str(self.params_path)
        config.model.enable_dynamic_shape = False
        config.model.dtype_override = DtypeOverride.fp32
        config.export.max_seq_length = 8
        config.export.max_context_length = 8
        config.export.output_dir = str(self.root)
        config.quantization.qmode = qmode
        return config

    def test_checkpoint_name_does_not_select_quantization(self):
        checkpoint_bytes = self.checkpoint_paths[0].read_bytes()
        baseline_types = None
        baseline_state = None

        for checkpoint_path in self.checkpoint_paths:
            with self.subTest(checkpoint_path=checkpoint_path):
                self.assertEqual(checkpoint_path.read_bytes(), checkpoint_bytes)
                model = Llama2Model(self._config(checkpoint_path)).get_eager_model()
                module_types = Counter(
                    type(module).__name__ for module in model.modules()
                )
                state = model.state_dict()

                self.assertEqual(module_types["WeightOnlyInt8Linear"], 0)
                if baseline_types is None:
                    baseline_types = module_types
                    baseline_state = state
                    continue

                self.assertEqual(module_types, baseline_types)
                self.assertEqual(state.keys(), baseline_state.keys())
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, baseline_state[name]), name)

    def test_explicit_quantization_is_independent_of_checkpoint_name(self):
        quantized_linear_counts = []
        for checkpoint_path in (
            self.checkpoint_paths[0],
            self.checkpoint_paths[-1],
        ):
            with self.subTest(checkpoint_path=checkpoint_path):
                manager = _prepare_for_llama_export(
                    self._config(checkpoint_path, qmode="int8")
                )
                module_types = Counter(
                    type(module).__name__ for module in manager.model.modules()
                )
                quantized_linear_counts.append(module_types["WeightOnlyInt8Linear"])
                self.assertGreater(module_types["WeightOnlyInt8Linear"], 0)

        self.assertEqual(quantized_linear_counts[0], quantized_linear_counts[1])


if __name__ == "__main__":
    unittest.main()
