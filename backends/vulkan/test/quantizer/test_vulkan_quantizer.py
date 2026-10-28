# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import torch
from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
    get_symmetric_quantization_config,
    VulkanQuantizer,
)
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class TestVulkanQuantizer(unittest.TestCase):
    def test_dynamic_int8_token_independence(self):
        torch.manual_seed(22431)
        model = torch.nn.Linear(32, 12).eval()
        x = torch.randn(2, 3, 32)
        quantizer = VulkanQuantizer().set_global(
            get_symmetric_quantization_config(is_dynamic=True, weight_bits=8)
        )
        prepared = prepare_pt2e(export(model, (x,)).module(), quantizer)
        prepared(x)
        converted = convert_pt2e(prepared)
        changed = x.clone()
        changed[1, 2] *= 1000
        torch.testing.assert_close(
            converted(x)[0], converted(changed)[0], rtol=0, atol=0
        )

    def test_int64_scalar_add_used_as_index(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x[:, torch.arange(4) + 0]

        quantizer = VulkanQuantizer()
        quantization_config = get_symmetric_quantization_config()
        quantizer.set_global(quantization_config)
        example_inputs = (torch.randn(1, 4, 5),)
        m = export(M(), example_inputs, strict=True).module()
        m = quantizer.transform_for_annotation(m)
        m = prepare_pt2e(m, quantizer)
        torch.testing.assert_close(m(*example_inputs), M()(*example_inputs))

    def test_int64_scalar_add_without_set_global(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x[:, torch.arange(4) + 0]

        example_inputs = (torch.randn(1, 4, 5),)
        m = export(M(), example_inputs, strict=True).module()
        m = VulkanQuantizer().transform_for_annotation(m)
        torch.testing.assert_close(m(*example_inputs), M()(*example_inputs))

    def test_scalar_type_promotion(self):
        class M(torch.nn.Module):
            def __init__(self, op, scalar):
                super().__init__()
                self.op = op
                self.scalar = scalar

            def forward(self, x):
                return self.op(x, self.scalar)

        cases = [
            (torch.float16, 1e-4, 100000.0),
            (torch.float16, 1e-4, 100000),
            (torch.float16, 10000.0, 1e-8),
            (torch.bfloat16, 100.0, 1.0039),
            (torch.float64, 1.0, 1.0 + 2**-30),
            (torch.int64, 2**54 + 1, 1),
            (torch.int32, 1, 1),
            (torch.int32, 1, 2**31),
            (torch.int8, 1, 256),
            (torch.bool, True, False),
            (torch.int32, 1, 0.5),
            (torch.float32, 1.0, 2),
            (torch.complex64, 1j, 1 + 2j),
        ]
        for op, (dtype, value, scalar), shape, configured in itertools.product(
            (torch.add, torch.mul), cases, ((), (2,)), (False, True)
        ):
            with self.subTest(
                op=op, dtype=dtype, scalar=scalar, shape=shape, configured=configured
            ):
                model = M(op, scalar)
                example_inputs = (torch.full(shape, value, dtype=dtype),)
                expected = model(*example_inputs)
                quantizer = VulkanQuantizer()
                if configured:
                    quantizer.set_global(get_symmetric_quantization_config())
                m = export(model, example_inputs, strict=True).module()
                m = prepare_pt2e(m, quantizer)
                torch.testing.assert_close(m(*example_inputs), expected, rtol=0, atol=0)

    def test_dynamic_int8_qparams_and_reference(self):
        for shape, bias, qrange in itertools.product(
            [(1, 4), (7, 32), (2, 3, 64), (2, 2, 3, 12)],
            [False, True],
            [(-128, 127), (-64, 63)],
        ):
            with self.subTest(shape=shape, bias=bias, qrange=qrange):
                torch.manual_seed(42)
                x = torch.randn(shape)
                rows = x.reshape(-1, shape[-1])
                rows[0].zero_()
                if rows.shape[0] > 1:
                    rows[1] = 1e-8
                    rows[-1] *= 100
                model = torch.nn.Linear(shape[-1], 13, bias=bias).eval()
                config = get_symmetric_quantization_config(
                    is_dynamic=True, act_qmin=qrange[0], act_qmax=qrange[1]
                )
                prepared = prepare_pt2e(
                    export(model, (x,)).module(), VulkanQuantizer().set_global(config)
                )
                prepared(x)
                converted = convert_pt2e(prepared)
                chooser = next(
                    n
                    for n in converted.graph.nodes
                    if n.op == "call_function"
                    and "choose_qparams_affine" in str(n.target)
                )
                self.assertEqual(
                    list(chooser.args[2]), [1] * (len(shape) - 1) + [shape[-1]]
                )
                scale, zp = torch.ops.torchao.choose_qparams_affine(
                    *((x,) + chooser.args[1:])
                )
                self.assertEqual(scale.numel(), x.numel() // x.shape[-1])
                self.assertEqual(zp.numel(), scale.numel())
                weight_dq = next(
                    n
                    for n in converted.graph.nodes
                    if n.op == "call_function"
                    and "dequantize_per_channel" in str(n.target)
                )
                weights = converted.get_buffer(weight_dq.args[0].target)
                weight_scales = converted.get_buffer(weight_dq.args[1].target)
                out = torch.ops.et_vk.linear_dq8ca_q8csw(
                    x,
                    weights,
                    weights.sum(1).int(),
                    weight_scales,
                    model.bias,
                    *qrange,
                )
                torch.testing.assert_close(out, converted(x), rtol=1e-5, atol=1e-5)

    def test_dynamic_legacy_and_weight_only_configs(self):
        for bits in (4, 8):
            self.assertIsNone(
                get_symmetric_quantization_config(weight_bits=bits).input_activation
            )
            legacy = get_symmetric_quantization_config(
                is_dynamic=True, weight_bits=bits, act_per_token=False
            )
            self.assertEqual(legacy.input_activation.qscheme, torch.per_tensor_affine)
        self.assertEqual(
            get_symmetric_quantization_config(
                is_dynamic=True, weight_bits=4
            ).input_activation.qscheme,
            torch.per_tensor_affine,
        )
