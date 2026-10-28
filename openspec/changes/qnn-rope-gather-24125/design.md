# Design

## Context

See proposal.md for motivation. The baseline is 390bc0618f and its QAIRT pin
is 2.51.0.260929; the reporter used a v1.4.1 fork and 2.47.0.260601. No device
was supplied. The repository documents Linux x64 HTP emulator execution.

## Goals / Non-Goals

**Goals:** Reproduce before modifying production code, localize the first
incorrect boundary, and verify a minimal fix with equivalent baseline/fixed runs.

**Non-Goals:** Unrelated backend refactors, SDK license acceptance, upstream
publication, and claiming emulator execution establishes physical HTPv81 acceptance.

## Decisions

Use the exact reporter graph, inputs, seed and calibration first. Keep debug
intermediate outputs disabled: APP_READ promotion can alter graph optimization.
Compare all returned tensors with eager and QDQ references. Use repository
download/build commands and qnn_executor_runner with libQnnHtp.so, never CPU
backend emulation as a substitute. Check SDK versions consistently at build,
export and execution. Compare the older SDK if compatible and obtainable.

Inspect index lowering, quantization attributes, layout, tensor wrapper reuse,
and runtime output binding. Do not insert an identity, copy or extra graph
output without evidence that it addresses the actual defect. If no failure
is reproduced, preserve diagnostics and explicitly leave fix acceptance blocked.

## Risks / Trade-offs

- Emulator/device differences → physical device acceptance remains separate.
- SDK version drift → record versions and rebuild for each supported comparison.
- Instrumentation changes graph optimization → original output-only model is the acceptance test.
- Limited workspace disk → SDK/download caches in /tmp; builds at most eight jobs.

## Verified correction

Both matched SDK builds reproduce k1 relative L2 0.7683357. Standalone SDK
qnn-net-run returns bitwise identical outputs, localizing the failure beyond
ExecuTorch runtime output handling. Consistent shape/layout/quantization metadata
does not explain it; the SDK's internal cause remains unproven.

Use GatherNd for axis-zero, rank-two UFIXED_POINT16 tables with vector indices.
Reshaping indices from [N] to [N,1] preserves indexing with batch_dims=0 and
avoids the failing Gather path. Original outputs reach QDQ accuracy on both SDKs.
Keep all other indexing forms unchanged. A single output-only RoPE also fails
before the change, so consumer count is deliberately not a condition. Do not
generalize this workaround to unrelated ranks/axes without further evidence.

## File changes

op_index.py implements the guarded equivalent lowering. tests/models.py adds
SharedGatherRoPE; test_qnn_delegate.py registers the original and smaller shapes
with eight calibration batches and all-output comparisons. Issue-scoped state,
acceptance report and OpenSpec artifacts record evidence and blocked physical
acceptance. No runtime, C++ build, dependencies or public APIs are changed.
