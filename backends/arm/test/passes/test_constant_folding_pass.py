# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch
from executorch.backends.arm._passes.constant_folding_pass import ConstantFoldingPass


class ReadOnlyBufferModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.ones(2, 1, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = cast(torch.Tensor, self.state)
        return x + state[0]


class MutableBufferModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.ones(2, 1, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = cast(torch.Tensor, self.state)
        state_0 = state[0]
        state_1 = state[1]
        new_state = torch.stack([state_0 + x, state_1 + x])
        state.copy_(new_state)
        return state_0 + x


def _export_module(module: torch.nn.Module) -> torch.fx.GraphModule:
    return torch.export.export(
        module.eval(),
        (torch.ones(1, 4),),
        strict=False,
    ).module()


def _contains_target(graph_module: torch.fx.GraphModule, target) -> bool:
    return any(
        node.op == "call_function" and node.target == target
        for node in graph_module.graph.nodes
    )


def test_constant_folding_folds_read_only_buffer() -> None:
    graph_module = _export_module(ReadOnlyBufferModel())

    result = ConstantFoldingPass()(graph_module)

    assert result is not None
    assert result.modified
    assert not _contains_target(graph_module, torch.ops.aten.select.int)
    assert any(
        name.startswith("_frozen_param") for name, _ in graph_module.named_buffers()
    )


def test_constant_folding_preserves_mutable_buffer_read() -> None:
    graph_module = _export_module(MutableBufferModel())

    result = ConstantFoldingPass()(graph_module)

    assert result is not None
    assert _contains_target(graph_module, torch.ops.aten.copy_.default)
    assert _contains_target(graph_module, torch.ops.aten.select.int)
    assert not any(
        name.startswith("_frozen_param") for name, _ in graph_module.named_buffers()
    )


class TrainableParameterModel(torch.nn.Module):
    """A per-channel scale, the shape QAT needs to keep trainable."""

    def __init__(self, num_channels: int = 3) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.scale = torch.nn.Parameter(torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1] * x.dim()
        shape[1] = self.num_channels
        return x * self.scale.view(shape)


def _export_trainable_module() -> torch.fx.GraphModule:
    return torch.export.export(
        TrainableParameterModel(),
        (torch.ones(1, 3, 2, 2),),
    ).module(check_guards=False)


def test_constant_folding_folds_trainable_parameter_by_default() -> None:
    graph_module = _export_trainable_module()

    result = ConstantFoldingPass()(graph_module)

    assert result is not None
    assert result.modified
    assert not _contains_target(graph_module, torch.ops.aten.view.default)


def test_constant_folding_preserves_trainable_parameter_when_asked() -> None:
    """Folding the view of a trainable parameter freezes it and drops it from
    model.parameters(), which leaves a QAT optimizer with an empty parameter
    list. See https://github.com/pytorch/executorch/issues/20148.
    """
    graph_module = _export_trainable_module()

    ConstantFoldingPass(preserve_trainable_parameters=True)(graph_module)

    assert _contains_target(graph_module, torch.ops.aten.view.default)
    assert len(list(graph_module.parameters())) == 1
