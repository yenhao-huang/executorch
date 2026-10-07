# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm.ethosu.compile_spec import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from torchao.quantization.pt2e import move_exported_model_to_train
from executorch.backends.arm.quantizer.arm_quantizer import (
    _any_config_uses_fake_quantization,
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e

NUM_CHANNELS = 3


class PerChannelScale(torch.nn.Module):
    """A per-channel scale, whose only parameter QAT has to keep trainable."""

    def __init__(self, num_channels: int = NUM_CHANNELS) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.scale = torch.nn.Parameter(torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1] * x.dim()
        shape[1] = self.num_channels
        return x * self.scale.view(shape)


def _make_quantizer(is_qat: bool) -> EthosUQuantizer:
    quantizer = EthosUQuantizer(EthosUCompileSpec(target="ethos-u55-128"))
    quantizer.set_global(get_symmetric_quantization_config(is_qat=is_qat))
    return quantizer


def _export(module: torch.nn.Module) -> torch.fx.GraphModule:
    return torch.export.export(
        module, (torch.randn(4, NUM_CHANNELS, 8, 8),)
    ).module(check_guards=False)


def test_qat_config_is_recognised_as_fake_quantization() -> None:
    assert _any_config_uses_fake_quantization(_make_quantizer(is_qat=True))
    assert not _any_config_uses_fake_quantization(_make_quantizer(is_qat=False))


def test_prepare_qat_keeps_trainable_parameters() -> None:
    """Constant folding used to fold the parameter's view, freezing it and
    leaving the QAT optimizer with an empty parameter list. See
    https://github.com/pytorch/executorch/issues/20148.
    """
    torch.manual_seed(0)
    prepared = prepare_qat_pt2e(_export(PerChannelScale()), _make_quantizer(True))

    assert len(list(prepared.parameters())) == 1


def test_prepare_qat_model_trains() -> None:
    torch.manual_seed(0)
    target = torch.tensor([2.7, 0.6, -1.4]).view(1, NUM_CHANNELS, 1, 1)
    prepared = move_exported_model_to_train(
        prepare_qat_pt2e(_export(PerChannelScale()), _make_quantizer(True))
    )
    optimizer = torch.optim.SGD(prepared.parameters(), lr=1.0, momentum=0.3)

    def step() -> float:
        x = torch.randn(16, NUM_CHANNELS, 8, 12)
        loss = torch.nn.functional.mse_loss(prepared(x), x * target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    first = step()
    for _ in range(7):
        last = step()

    assert last < first
    scale = next(iter(prepared.parameters())).detach().flatten()
    torch.testing.assert_close(scale, target.flatten(), atol=0.2, rtol=0.2)


def test_prepare_ptq_still_folds() -> None:
    torch.manual_seed(0)
    prepared = prepare_pt2e(_export(PerChannelScale()), _make_quantizer(False))

    assert len(list(prepared.parameters())) == 0
