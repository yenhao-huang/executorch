# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.pass_base import ExportPass, PassResult
from torchao.quantization.pt2e.constant_fold import constant_fold


class ConstantFoldingPass(ArmPass):
    """Fold constant subgraphs using torch's export constant folding pass.

    To be used before to_edge transform.

    Args:
        preserve_trainable_parameters (bool): Leave subgraphs that read a
            parameter with ``requires_grad`` alone. Folding such a subgraph
            freezes the parameter at its current value and drops it from
            ``model.parameters()``, which breaks a QAT loop; set this when
            preparing for QAT.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    parameter_targets = {
        torch.ops.aten.linear.default,
        torch.ops.aten.convolution.default,
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv1d.padding,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv2d.padding,
        torch.ops.aten.conv3d.default,
        torch.ops.aten.conv3d.padding,
        torch.ops.aten.conv_transpose2d.input,
    }

    def __init__(self, *args, preserve_trainable_parameters: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.preserve_trainable_parameters = preserve_trainable_parameters

    def _reads_trainable_parameter(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node
    ) -> bool:
        """Whether `node`'s value is computed from a parameter being trained."""
        seen: Set[torch.fx.Node] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current.op == "get_attr":
                attr = getattr(graph_module, current.target, None)  # type: ignore[arg-type]
                if isinstance(attr, torch.nn.Parameter) and attr.requires_grad:
                    return True
                continue
            stack.extend(
                arg for arg in current.all_input_nodes if isinstance(arg, torch.fx.Node)
            )
        return False

    def _ensure_param_attr(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node | None
    ) -> bool:
        """Replaces tensor attributes with parameter attributes."""
        if node is None or node.op != "get_attr":
            return False
        target = node.target
        try:
            attr = getattr(graph_module, target)  # type: ignore[arg-type]
        except AttributeError:
            return False
        if isinstance(attr, torch.nn.Parameter):
            return False
        if not isinstance(attr, torch.Tensor):
            return False
        setattr(graph_module, target, torch.nn.Parameter(attr, requires_grad=False))  # type: ignore[arg-type]
        return True

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        before_graph = str(graph_module.graph)
        constraint_fn = None
        if self.preserve_trainable_parameters:

            def constraint_fn(node: torch.fx.Node) -> bool:
                return not self._reads_trainable_parameter(graph_module, node)

        constant_fold(graph_module, constraint_fn)

        modified = before_graph != str(graph_module.graph)

        # Constant folding may have replaced some parameters with tensors, so we need to ensure they are still parameters for the backend to work correctly.
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in self.parameter_targets:
                args = node.args
                if len(args) > 1:
                    modified |= self._ensure_param_attr(graph_module, args[1])
                if len(args) > 2:
                    modified |= self._ensure_param_attr(graph_module, args[2])

        return PassResult(graph_module, modified)
