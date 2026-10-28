# Design

## Context

See proposal.md for motivation. The baseline is 390bc0618f. Existing dynamic INT4 code assumes effectively two-dimensional activations and can silently select weight-only computation without integer dot product support. The new INT8 path must preserve activation quantization on such devices and cover batches.

## Goals / Non-Goals

**Goals:** correct per-token host conversion, fused reference, graph recognition, Vulkan quantization and INT32 accumulation; explicit legacy per-tensor selection.

**Non-Goals:** repairing the separate static activation issue, altering INT4 numerics, or adding a new cooperative matrix optimization.

## Decisions

Use torchao's AffineQuantizedPlaceholderObserver with PerToken and asymmetric INT8 mapping. Default dynamic INT8 weights to per-token; preserve INT4's default and add an explicit activation granularity switch. This uses torchao's supported conversion hook rather than relying on a qscheme to imply token granularity.

Fuse only a fully verified per-token affine Q/DQ pattern with symmetric per-channel INT8 weights. The new linear_dq8ca_q8csw op computes token qparams internally; dead-code elimination removes the original chooser only when no other consumers remain. Its reference explicitly quantizes and dequantizes activations before linear. This avoids transporting/reinterpreting per-token qparams across incompatible layouts.

Use contiguous buffers for activation input/output and an internal packed INT8 activation buffer, plus the existing INT8 weight prepacking format. A row reduction/quantization shader computes qparams and packed activations for all leading dimensions flattened into rows. A matrix shader accumulates packed INT8 values in INT32, subtracts activation zero-point times precomputed weight sums, applies scales and bias. A Vulkan shader variant uses explicit integer arithmetic on devices lacking integer dot products; it never drops activation quantization. The initial runtime implementation supports float32 inputs and qparams.

## Risks / Trade-offs

- Rounding and small-scale behavior → use the same affine formula, epsilon, and round-to-even in reference and shaders; test constant, zero, asymmetric, and tiny rows.
- Shape/layout mismatch → require contiguous buffers, use actual runtime row counts, and test multiple batches and dynamic tokens.
- Fusing unsupported qparams → verify chooser arguments, block sizes, dtype, zero-point domain, and matching Q/DQ nodes.
- Performance → baseline implementation prioritizes a correct portable Vulkan path; measure latency without claiming universal speedups.

## Planned files

Modify quantizer/vulkan_quantizer.py, patterns/quantized_linear.py, custom_ops_lib.py, op_registry.py and quantization documentation. Add runtime/graph/ops/impl/gemm/linear/qw/dq8ca_q8csw and matching GLSL/YAML. Extend registered Python quantizer and Vulkan delegate tests. Track actual changes and validation in docs/howard/22431.md.
