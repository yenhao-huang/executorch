# Spec Delta

## Purpose

Preserve token-level activation fidelity when executing dynamically quantized INT8 linear layers on Vulkan.

## ADDED Requirements

### Requirement: Token-local activation quantization
Dynamic INT8 activation and INT8 weight configuration SHALL compute independent asymmetric activation qparams over each token's final feature dimension. An explicit per-tensor compatibility setting SHALL remain available.

#### Scenario: Different token magnitudes
- **WHEN** independent tokens have greatly different ranges
- **THEN** changing one token SHALL NOT change another token's quantized linear output

### Requirement: Matching host and Vulkan semantics
The converted graph, fused reference, and Vulkan execution SHALL apply activation rounding, clamping, zero points, per-output-channel INT8 weight scales, and optional bias consistently.

#### Scenario: Batched inference
- **WHEN** supported batched or unbatched inputs with optional bias execute
- **THEN** Vulkan results SHALL agree with the quantized reference within documented numerical tolerances

#### Scenario: Devices without integer dot product
- **WHEN** a Vulkan device lacks the integer dot product extension
- **THEN** activation quantization SHALL still execute on Vulkan and retain its specified numerical behavior

#### Scenario: Forced half storage
- **WHEN** a supported float32 graph is lowered with force_fp16 enabled
- **THEN** the runtime SHALL use half storage for activations, weight scales, bias and outputs, float32 activation qparams and scale arithmetic, and INT32 accumulation
- **AND** results SHALL agree with a reference that accounts for half storage rounding

#### Scenario: Dynamic token counts exceeding dispatch limits
- **WHEN** a supported input grows beyond the device's row workgroup limit
- **THEN** every token SHALL execute with independent qparams without exceeding dispatch limits or requiring standalone qparam textures

#### Scenario: Multiple output consumers
- **WHEN** a dynamically shaped linear output feeds multiple operators
- **THEN** runtime resizing SHALL preserve the reference results for every consumer

### Requirement: Compatibility
Existing weight-only INT8, INT4, and static activation paths SHALL retain their behavior.

#### Scenario: Adjacent configurations
- **WHEN** existing quantization configurations are selected
- **THEN** relevant regression tests SHALL preserve their prior results, with pre-existing failures documented separately
