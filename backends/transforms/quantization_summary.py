# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A post-hoc summary of what a quantizer annotated.

Quantization can "succeed" and do nothing: an unsupported pattern, a filter
that matches no module, or a config that never applies all leave the graph
untouched, and the only evidence is the exported program itself, which is long
and hard to read at a glance. This reads the annotations a quantizer left on a
graph and logs a short summary, so the common failure is visible.

    Quantization summary (XNNPACKQuantizer): 10 of 42 nodes annotated
      linear: 10 instances, static, act int8 per-tensor affine, weight int8 per-channel symmetric

Unlike backends/cortex_m/quantizer_reporter.py, which a quantizer instruments
by reporting each accepted and rejected pattern as it annotates, this needs no
cooperation from the quantizer: it reads the finished annotations.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, NamedTuple, Optional

import torch
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

logger = logging.getLogger(__name__)

_DTYPE_BITS: Dict[torch.dtype, int] = {
    torch.int8: 8,
    torch.uint8: 8,
    torch.int16: 16,
    torch.uint16: 16,
    torch.int32: 32,
    torch.uint32: 32,
    torch.float8_e4m3fn: 8,
    torch.float8_e5m2: 8,
}


def _bit_width(spec: QuantizationSpec) -> Optional[int]:
    """The width a spec quantizes to, from its range when it narrows one."""
    if spec.quant_min is not None and spec.quant_max is not None:
        levels = spec.quant_max - spec.quant_min + 1
        if levels > 1:
            # A range of 2**n levels, or 2**n - 1 for a symmetric signed range.
            for bits in range(1, 33):
                if levels <= (1 << bits):
                    return bits
    return _DTYPE_BITS.get(spec.dtype)


def _granularity(spec: QuantizationSpec) -> str:
    qscheme = spec.qscheme
    if qscheme is None:
        return "unknown-granularity"
    name = str(qscheme).removeprefix("torch.")
    granularity = "per-channel" if "per_channel" in name else "per-tensor"
    symmetry = "symmetric" if "symmetric" in name else "affine"
    return f"{granularity} {symmetry}"


def describe_spec(spec: Optional[QuantizationSpecBase]) -> str:
    """A short, human-readable description of one quantization spec."""
    if spec is None:
        return "none"
    if isinstance(spec, SharedQuantizationSpec):
        return "shared"
    if isinstance(spec, DerivedQuantizationSpec):
        return "derived"
    if not isinstance(spec, QuantizationSpec):
        return spec.__class__.__name__

    bits = _bit_width(spec)
    dtype = str(spec.dtype).removeprefix("torch.")
    width = f"int{bits}" if bits is not None and "int" in dtype else dtype
    return f"{width} {_granularity(spec)}"


def _is_weight_input(node: Node, spec: Optional[QuantizationSpecBase]) -> bool:
    """Whether an entry of input_qspec_map describes a weight.

    A per-channel spec is always a weight spec, and otherwise a constant input
    is a weight while a computed one is an activation.
    """
    if isinstance(spec, QuantizationSpec) and spec.qscheme is not None:
        if "per_channel" in str(spec.qscheme):
            return True
    return node.op == "get_attr"


def describe_annotation(node: Node, annotation: QuantizationAnnotation) -> str:
    """A short description of the scheme a node was annotated with."""
    activation_specs: List[str] = []
    weight_specs: List[str] = []
    dynamic = False

    for input_node, spec in annotation.input_qspec_map.items():
        if spec is None:
            continue
        if isinstance(spec, QuantizationSpec) and spec.is_dynamic:
            dynamic = True
        description = describe_spec(spec)
        if _is_weight_input(input_node, spec):
            weight_specs.append(description)
        else:
            activation_specs.append(description)

    parts = ["dynamic" if dynamic else "static"]
    if activation_specs:
        parts.append(f"act {sorted(set(activation_specs))[0]}")
    if weight_specs:
        parts.append(f"weight {sorted(set(weight_specs))[0]}")
    if annotation.output_qspec is not None:
        parts.append(f"out {describe_spec(annotation.output_qspec)}")
    return ", ".join(parts)


def node_label(node: Node) -> str:
    """The name to group a node under.

    The innermost module the node came from, which is the level a user thinks
    in, or the op name when the node does not sit inside a submodule — the
    whole model is the outermost stack entry and says nothing useful.
    """
    module_stack = node.meta.get("nn_module_stack")
    if module_stack and len(module_stack) > 1:
        _, innermost = list(module_stack.values())[-1]
        if isinstance(innermost, type):
            return innermost.__name__
        if isinstance(innermost, str):
            return innermost.rsplit(".", 1)[-1]
    target = node.target
    name = getattr(target, "_opname", None) or getattr(target, "__name__", None)
    return name or str(target)


class QuantizationSummary(NamedTuple):
    """Counts of what a quantizer annotated, grouped by op and scheme."""

    # (label, scheme) -> number of nodes
    counts: Counter
    annotated_nodes: int
    total_nodes: int

    def lines(self) -> List[str]:
        """The summary as one line per (op, scheme) group, most common first."""
        return [
            f"{label}: {count} instance{'s' if count != 1 else ''}, {scheme}"
            for (label, scheme), count in self.counts.most_common()
        ]


def summarize_quantization(model: GraphModule) -> QuantizationSummary:
    """Read the quantization annotations a quantizer left on `model`."""
    counts: Counter = Counter()
    annotated = 0
    total = 0

    for node in model.graph.nodes:
        if node.op != "call_function":
            continue
        total += 1
        annotation = node.meta.get(Q_ANNOTATION_KEY)
        if annotation is None or not annotation._annotated:
            continue
        if not annotation.input_qspec_map and annotation.output_qspec is None:
            continue
        annotated += 1
        counts[(node_label(node), describe_annotation(node, annotation))] += 1

    return QuantizationSummary(counts, annotated, total)


def log_quantization_summary(model: GraphModule, quantizer_name: str) -> None:
    """Log what `quantizer_name` annotated on `model`.

    A summary at INFO, a per-node listing at DEBUG, and a warning when nothing
    was annotated at all, which is the failure that otherwise looks like
    success.
    """
    summary = summarize_quantization(model)

    if summary.annotated_nodes == 0:
        logger.warning(
            "%s did not find any operators to quantize; the model will come out "
            "unquantized. Check that the configured patterns and filters match "
            "this model.",
            quantizer_name,
        )
        return

    logger.info(
        "Quantization summary (%s): %d of %d nodes annotated",
        quantizer_name,
        summary.annotated_nodes,
        summary.total_nodes,
    )
    for line in summary.lines():
        logger.info("  %s", line)

    if logger.isEnabledFor(logging.DEBUG):
        for node in model.graph.nodes:
            annotation = node.meta.get(Q_ANNOTATION_KEY)
            if annotation is None or not annotation._annotated:
                continue
            logger.debug(
                "  %s (%s): %s",
                node.name,
                node_label(node),
                describe_annotation(node, annotation),
            )
