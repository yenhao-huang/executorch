# Proposal

## Why

The reported QNN 16a8w multi-RoPE model returns an inaccurate output-only key
tensor after sharing gathered cos/sin. Establish the failure on the requested
main baseline and fix only a demonstrated defect.

Related issue: [ExecuTorch 24125](https://redirect.github.com/pytorch/executorch/issues/24125)

## What Changes

- Reproduce on the documented Linux x64 HTP emulator with consistent SDK versions.
- Investigate lowering, tensor metadata and runtime output handling.
- Add focused regression coverage and a minimal production fix only if evidence supports one.
- Record blocked physical-device acceptance and baseline/fixed evidence.

## Capabilities

### New Capabilities

- `qnn-shared-gather`: Numerical correctness when gathered tensors feed multiple consumers including graph outputs.

### Modified Capabilities

None.

## Impact

Candidate areas are Qualcomm builders, passes, tensor wrappers and runtime;
specific code edits depend on reproduction. Required records are
`docs/goals/issue-24125/STATE.md` and `docs/howard/24125.md`. No new dependency
or public API change is planned.
