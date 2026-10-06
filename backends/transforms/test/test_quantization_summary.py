# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import torch

from executorch.backends.transforms.quantization_summary import (
    log_quantization_summary,
    summarize_quantization,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e


class ConvLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3)
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).mean(dim=(2, 3))
        return self.fc2(torch.relu(self.fc1(x)))

    def example_inputs(self):
        return (torch.randn(2, 3, 8, 8),)


def _annotated_model(quantizer: XNNPACKQuantizer) -> torch.fx.GraphModule:
    model = ConvLinear().eval()
    inputs = model.example_inputs()
    prepared = prepare_pt2e(torch.export.export(model, inputs).module(), quantizer)
    prepared(*inputs)
    return prepared


class QuantizationSummaryTest(unittest.TestCase):
    """Quantization can "succeed" and do nothing, which the summary makes
    visible. See https://github.com/pytorch/executorch/issues/7718.
    """

    def test_summary_counts_and_describes_what_was_annotated(self):
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        summary = summarize_quantization(_annotated_model(quantizer))

        self.assertGreater(summary.annotated_nodes, 0)
        self.assertLessEqual(summary.annotated_nodes, summary.total_nodes)

        lines = "\n".join(summary.lines())
        self.assertIn("Linear", lines)
        self.assertIn("Conv2d", lines)
        self.assertIn("static", lines)
        self.assertIn("weight int8 per-channel symmetric", lines)

        labels = {label for label, _ in summary.counts}
        self.assertEqual(
            sum(summary.counts.values()),
            summary.annotated_nodes,
            "every annotated node should be counted once",
        )
        # The whole model is the outermost module and says nothing useful, so a
        # node outside a submodule is labelled by its op.
        self.assertNotIn("ConvLinear", labels)

    def test_dynamic_activation_is_reported_as_dynamic(self):
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
        )
        summary = summarize_quantization(_annotated_model(quantizer))

        lines = "\n".join(summary.lines())
        self.assertIn("dynamic", lines)
        self.assertNotIn("static", lines)

    def test_nothing_annotated_warns(self):
        # A quantizer with no config annotates nothing, and the export still
        # "succeeds".
        quantizer = XNNPACKQuantizer()
        model = ConvLinear().eval()
        graph_module = torch.export.export(model, model.example_inputs()).module()

        summary = summarize_quantization(graph_module)
        self.assertEqual(summary.annotated_nodes, 0)

        with self.assertLogs(
            "executorch.backends.transforms.quantization_summary",
            level=logging.WARNING,
        ) as logs:
            log_quantization_summary(graph_module, "TestQuantizer")
        self.assertIn("did not find any operators to quantize", "\n".join(logs.output))

    def test_summary_is_logged_during_annotation(self):
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        model = ConvLinear().eval()
        exported = torch.export.export(model, model.example_inputs()).module()

        with self.assertLogs(
            "executorch.backends.transforms.quantization_summary",
            level=logging.INFO,
        ) as logs:
            prepare_pt2e(exported, quantizer)
        output = "\n".join(logs.output)
        self.assertIn("Quantization summary (XNNPACKQuantizer)", output)
        self.assertIn("Linear", output)


if __name__ == "__main__":
    unittest.main()
