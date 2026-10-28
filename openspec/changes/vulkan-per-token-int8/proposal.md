# Proposal

## Why

Dynamic per-tensor activation quantization loses embedding fidelity on the encoder models reported in [ExecuTorch issue 22431](https://redirect.github.com/pytorch/executorch/issues/22431). Vulkan needs a per-token activation path with INT8 per-channel weights so users can retain dynamic quantization without sharing one scale across all tokens.

## What Changes

- Add an explicit per-token dynamic INT8 activation configuration for Vulkan linear quantization, with a documented compatibility policy for the existing per-tensor setting.
- Lower the supported per-token activation and INT8 weight pattern to a custom op with matching eager reference, Vulkan runtime and shader behavior.
- Test token-specific qparams, batch/shape/bias behavior, actual Vulkan execution and adjacent existing quantized paths.
- Compare encoder fidelity and latency with the existing per-tensor path where model and hardware prerequisites permit, recording all unavailable checks.

## Capabilities

### New Capabilities

- `vulkan-per-token-int8-linear`: Quantization, lowering and Vulkan execution of linear layers using per-token dynamic INT8 activations and per-output-channel INT8 weights.

### Modified Capabilities

None. No existing OpenSpec capabilities were present in this worktree.

## Impact

Changes span backends/vulkan/quantizer, patterns/quantized_linear.py, custom_ops_lib.py, op_registry.py, runtime/graph/ops/impl/gemm/linear/qw, corresponding GLSL registration and tests. Existing per-tensor/static, weight-only and INT4 behavior must remain available. The earlier comment and scalar-dtype fixes are already merged and are outside this change. No community writes or unrelated QNN modifications are included. Fork delivery targets yenhao-huang/executorch.
