# Copyright © 2024 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

from typing import Tuple

import numpy as np
import pytest

import torch

from coremltools.optimize.torch.quantization.quantization_config import (
    LinearQuantizerConfig,
    QuantizationScheme,
)

from executorch.backends.apple.coreml.quantizer import CoreMLQuantizer
from executorch.backends.apple.coreml.quantizer._annotation_config import (
    AnnotationConfig,
)
from torch.export import export

from torchao.quantization.pt2e.fake_quantize import FakeQuantizeBase
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)


def _get_quantization_config():
    return LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": QuantizationScheme.symmetric,
                "milestones": [0, 0, 10, 10],
                "activation_dtype": torch.quint8,
                "weight_dtype": torch.qint8,
                "weight_per_channel": True,
            }
        }
    )


def _count_fake_quantize_modules(model: torch.nn.Module) -> int:
    """Count the number of FakeQuantizeBase modules in a model."""
    count = 0
    for module in model.modules():
        if isinstance(module, FakeQuantizeBase):
            count += 1
    return count


class TestCoreMLQuantizer:
    @staticmethod
    def quantize_and_compare(
        model,
        example_inputs: Tuple[torch.Tensor],
        quantization_type: str,
    ) -> None:
        assert quantization_type in {"PTQ", "QAT"}

        pre_autograd_aten_dialect = export(model, example_inputs, strict=True).module()

        quantization_config = _get_quantization_config()
        quantizer = CoreMLQuantizer(quantization_config)

        if quantization_type == "PTQ":
            prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
        elif quantization_type == "QAT":
            prepared_graph = prepare_qat_pt2e(pre_autograd_aten_dialect, quantizer)
        else:
            raise ValueError("Invalid quantization type")

        print("Prepared graph:", prepared_graph)
        prepared_graph(*example_inputs)
        converted_graph = convert_pt2e(prepared_graph)
        print("Converted graph:", converted_graph)

        model_output = model(*example_inputs).detach().numpy()
        quantized_output = converted_graph(*example_inputs).detach().numpy()
        np.testing.assert_allclose(quantized_output, model_output, rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize("quantization_type", ("PTQ", "QAT"))
    def test_fake_quantize_modules_inserted_after_prepare(self, quantization_type):
        """Test that FakeQuantizeBase modules are inserted after prepare step."""
        SHAPE = (1, 3, 256, 256)

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, padding=1
                )
                self.relu = torch.nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a = self.conv(x)
                return self.relu(a)

        model = Model()
        example_inputs = (torch.randn(SHAPE),)

        pre_autograd_aten_dialect = export(model, example_inputs, strict=True).module()

        # Verify no FakeQuantize modules before prepare
        assert _count_fake_quantize_modules(pre_autograd_aten_dialect) == 0

        quantization_config = _get_quantization_config()
        quantizer = CoreMLQuantizer(quantization_config)

        if quantization_type == "PTQ":
            prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
        else:
            prepared_graph = prepare_qat_pt2e(pre_autograd_aten_dialect, quantizer)

        # Verify FakeQuantize modules are present after prepare
        fake_quant_count = _count_fake_quantize_modules(prepared_graph)
        assert fake_quant_count > 0, (
            f"Expected FakeQuantizeBase modules after prepare_{quantization_type.lower()}_pt2e, "
            f"but found {fake_quant_count}"
        )

    @pytest.mark.parametrize("quantization_type", ("PTQ", "QAT"))
    def test_conv_relu(self, quantization_type):
        SHAPE = (1, 3, 256, 256)

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, padding=1
                )
                self.relu = torch.nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a = self.conv(x)
                return self.relu(a)

        model = Model()

        example_inputs = (torch.randn(SHAPE),)
        self.quantize_and_compare(
            model,
            example_inputs,
            quantization_type,
        )

    @pytest.mark.parametrize("quantization_type", ("PTQ", "QAT"))
    def test_linear(self, quantization_type):
        SHAPE = (1, 5)

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = Model()

        example_inputs = (torch.randn(SHAPE),)
        self.quantize_and_compare(
            model,
            example_inputs,
            quantization_type,
        )


class TestCoreMLQuantizerWeightBits:
    """A sub-byte weight_dtype such as "qint4" reaches ExecuTorch as an int8
    dtype plus weight_n_bits=4, so reading only the dtype silently produced
    8-bit weights. See https://github.com/pytorch/executorch/issues/9649.
    """

    @staticmethod
    def _config(weight_dtype, weight_per_channel: bool = True):
        return LinearQuantizerConfig.from_dict(
            {
                "global_config": {
                    "quantization_scheme": QuantizationScheme.symmetric,
                    "activation_dtype": torch.float32,
                    "weight_dtype": weight_dtype,
                    "weight_per_channel": weight_per_channel,
                }
            }
        )

    @pytest.mark.parametrize(
        "weight_dtype,expected_range",
        [
            ("qint4", (-8, 7)),
            ("quint4", (0, 15)),
            (torch.qint8, (-128, 127)),
            (torch.quint8, (0, 255)),
        ],
    )
    def test_weight_quant_range_follows_n_bits(self, weight_dtype, expected_range):
        annotation_config = AnnotationConfig.from_quantization_config(
            self._config(weight_dtype).global_config
        )
        weight = annotation_config.weight
        assert (weight.quant_min, weight.quant_max) == expected_range

    @pytest.mark.parametrize("weight_per_channel", [True, False])
    def test_four_bit_weights_use_sixteen_levels(self, weight_per_channel):
        class Linear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(32, 8, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        torch.manual_seed(0)
        model = Linear().eval()
        inputs = (torch.randn(4, 32),)

        quantizer = CoreMLQuantizer(self._config("qint4", weight_per_channel))
        prepared = prepare_pt2e(export(model, inputs).module(), quantizer)
        prepared(*inputs)
        converted = convert_pt2e(prepared)

        int_weights = None
        graph_range = None
        buffers = dict(converted.named_buffers())
        for node in converted.graph.nodes:
            if node.op == "call_function" and "dequantize_per" in str(node.target):
                source = node.args[0]
                int_weights = buffers.get(source.target, None)
                if int_weights is None:
                    int_weights = getattr(converted, source.target)
                graph_range = (node.args[-3], node.args[-2])
                break

        assert int_weights is not None, "no dequantize node in the converted graph"
        assert graph_range == (-8, 7)
        assert int_weights.min() >= -8
        assert int_weights.max() <= 7
        # 4 bits is 16 levels; the 8-bit path uses far more than that.
        assert len(torch.unique(int_weights)) <= 16


if __name__ == "__main__":
    test_runner = TestCoreMLQuantizer()
    test_runner.test_conv_relu("PTQ")
    test_runner.test_linear("QAT")
    test_runner.test_fake_quantize_modules_inserted_after_prepare("PTQ")
