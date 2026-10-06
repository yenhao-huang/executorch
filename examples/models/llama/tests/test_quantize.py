# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from executorch.examples.models.llama.source_transformation.quantize import quantize


class QuantizeReportingTest(unittest.TestCase):
    """A group-wise scheme cannot cover a row that the group size does not
    divide, and those layers used to be dropped without a word. See
    https://github.com/pytorch/executorch/issues/22044.
    """

    class TwoLinears(nn.Module):
        def __init__(self):
            super().__init__()
            self.wide = nn.Linear(512, 512, bias=False)
            self.narrow = nn.Linear(576, 576, bias=False)

        def forward(self, x):
            return self.narrow(self.wide(x))

    def test_skipped_layers_are_reported(self):
        model = self.TwoLinears().eval()
        with self.assertLogs(level="WARNING") as logs:
            quantized = quantize(model, "8da4w", group_size=128)

        messages = "\n".join(logs.output)
        self.assertIn("narrow", messages)
        self.assertIn("in_features=576", messages)
        self.assertIn("group_size 128", messages)

        # 576 is not a multiple of 128, so that layer stays in fp32 while the
        # 512-wide one is quantized.
        # torchao replaces a quantized weight with a tensor subclass, so a
        # plain nn.Parameter means the layer was left alone.
        self.assertIs(type(quantized.narrow.weight), nn.Parameter)
        self.assertIsNot(type(quantized.wide.weight), nn.Parameter)

    def test_group_size_that_divides_every_layer_skips_nothing(self):
        model = self.TwoLinears().eval()
        with self.assertLogs(level="INFO") as logs:
            quantized = quantize(model, "8da4w", group_size=64)

        self.assertIn("skipped 0", "\n".join(logs.output))
        self.assertIsNot(type(quantized.narrow.weight), nn.Parameter)
        self.assertIsNot(type(quantized.wide.weight), nn.Parameter)

    def test_8da8w_quantizes(self):
        model = nn.Sequential(nn.Linear(512, 512, bias=False)).eval()
        quantized = quantize(model, "8da8w", group_size=128)
        self.assertIsNot(type(quantized[0].weight), nn.Parameter)


if __name__ == "__main__":
    unittest.main()
