# Solve Issue State

Run ID: issue-22431
Instance: et-04-22431 (existing tmux worker; no nested agents)
Started: 2026-10-28
Scope: per-token dynamic INT8 activations plus INT8 weights on Vulkan
Repository: yenhao-huang/executorch (fork writes only)
Issue: https://redirect.github.com/pytorch/executorch/issues/22431
Branch: agent/fix-22431
Baseline: 390bc0618f19276e699a255bd035a35c5296cfa8
Pull request: https://github.com/yenhao-huang/executorch/pull/46 (OPEN, non-draft)
Initial implementation commit: 5bb3afb4119823d75e6af9a8d9d60661516226e3
Prior FP16/dispatch correction: 7f295b497a7aad79130ce23e0124a6427b145001
Prior reviewed head: 8c8043807f5de3106a5a1a7d3329a0fc1b1046b7
Prior shared-activation correction: f0ba929cb08c15f9ee851ae716a830e34433c8b3
Round-three reviewed head: e6aac4f9cc1d1094d67ca3ee20e7b2a6868a13f3
Current correction commit: 64df469e6e1acbed8cd527e0173e8d41a741a3f2 (round-three allowlist compatibility; fork head verified)
Fork PR base: base/upstream-main-390bc0618f at 390bc0618f19276e699a255bd035a35c5296cfa8
Outcome: in_progress — round-three correction pushed and verified; parent independent review pending
OpenSpec: openspec/changes/vulkan-per-token-int8
Report: docs/howard/22431.md
Logs: ../logs/

Last updated: 2026-10-29

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Confirm Repository And Issue | completed | branch/HEAD checked | Exclusive worktree ownership from parent |
| 1. Check Prior Discussion For Existing Resolution | completed | issue-current.json, issue-comments.json, issue-timeline.json, pr-22432.json, pr-22372.json | Open, no comments; linked corrections do not resolve feature |
| 2. Read Issue And Repository Context | completed | required skills, AGENTS.md, CONTRIBUTING.md, OpenSpec proposal/design/spec | No community writes |
| 3. Implement And Test | completed | allowlist-correction/registered-before.log, targeted-final.log, full-regression.log | Two allowlist regressions fail before; five targeted tests and 72 broad Vulkan/WebGPU tests pass after. |
| 4. Review And Commit | completed | allowlist-correction/commit.log, lint-final.log, openspec-validate.log | P1 correction 64df469e6e committed; P3 full-pipeline evidence recorded; lint, hook and OpenSpec pass. |
| 5. Open And Follow Through On PR | in_progress | allowlist-correction/fork-code-readback.txt, pr-code-readback.json | Current correction 64df469e6e pushed normally and PR 46 head verified; parent review pending. |
| 6. Report Result | in_progress | docs/howard/22431.md | Review evidence updated; independent acceptance pending; physical-GPU/encoder limits retained. |

Setup limitation: Eigen GitLab clone repeatedly fails remotely; retry or use build configuration that does not require optimized kernels. No security settings changed.

Milestone 2026-10-28 16:34 UTC: baseline token-independence test failed (36/36 mismatches, max abs 0.7333376); fixed test passed. Quantizer suite 6 passed; Vulkan passes suite 34 passed; serialized runtime 2 tests passed (16 shape/bias/output-width combinations plus repeated calls and dynamic token sizes). C++ IntegerDotAndFallback passed 12 cases; standalone CTest discovers zero tests, so the standard GTest target was executed directly. Logs: baseline-regression.log, quantizer-tests.log, passes-tests-2.log, vulkan-runtime-tests.log, cpp-runtime-tests.log.

Optional encoder measurements: blocked; required existing HF CLI environment and model root absent. download-models runtime instructions prohibit replacement without separate request. No model files downloaded.

Milestone 2026-10-28: expanded serialized Vulkan regression passed 40 shape/output-width/bias/range combinations, rank-one through rank-four, plus dynamic tokens. Shared-weight sequence passed. C++ test passed 16 cases including texture and buffer weights and dot/no-dot variants. Static tests pass baseline/fixed; enabled INT4 test has identical seven tolerance mismatches on both. Temporary decorators/API edits restored. Fork main is 449 commits behind baseline; use a new fork-only base/upstream-main-390bc0618f snapshot to keep the PR isolated.

Initial delivery milestone (historical): implementation commit 5bb3afb4119823d75e6af9a8d9d60661516226e3 verified on fork branch agent/fix-22431; fork PR 46 verified OPEN, non-draft, based on the exact requested baseline. No checks or reviews reported. Community timeline readback contains no fork cross-reference. No merge, force push, deletion, community writes or nested workers occurred. Final report and completion record are published in a documentation follow-up commit.

## Claude review revision, 2026-10-29

Reviewed head 00f4af4228501c60ca97b4cf036d3d0ae459013e. Read parent review and reproduced both confirmed bugs with registered pytest cases: force_fp16 fails kFloat assertion; 65536 tokens exceeds workgroup limits (2 failed, 1 multi-consumer dynamic test passed; logs/claude-review/registered-before.log). Implemented typed half shader variants with FP32 qparams/arithmetic and bounded grid-stride row loops. Required workflow docs retained. Remaining tentative rank-one matching scope narrowed to per-channel INT8 weights. Claude acceptance remains pending parent review.

Revision milestone: force_fp16 uses typed half storage and float32 qparams; both
shaders bound row dispatch and loop across excess tokens. Dynamic large-token
fusion now consumes internal qparams without imposing standalone texture limits.
Regression before/fixed: 2 failures plus 1 pass became 3 passes. Full Python
selection: 50 passed. C++: 32 INT8 cases plus existing INT4 GEMM passed. Existing
static passed; enabled INT4 retains identical seven baseline mismatches. Broader
C++ INT4 missing-shader and sporadic CPU-reference failures reproduce with
baseline test source; detailed limits and logs are in the report. Required
workflow docs remain. Parent owns the next Claude pass and final acceptance.

Review handoff: implementation revision 7f295b497a verified on fork PR 46; no GitHub checks or reviews are reported. This documentation follow-up records delivery and remains part of the next parent-reviewed head. Final remote readback is logs/claude-review/pr-final-readback.json under the task parent. No merge or community writes occurred.


## Shared activation correction

Parent disproved round-two Claude's assumption that PT2E cannot share activation
observers. At head 8c8043807f, the registered two-projection test verifies one
chooser/Q/DQ with two dequantize consumers; static execution passes, dynamic
max=65540 fails with Missing out variants: torchao::choose_qparams_affine
(shared-activation-review/registered-before.log). This is a gap in the new feature;
the legacy per-tensor control also fails, so it is not a demonstrated legacy
regression. Do not treat round-two acceptance as acceptance of the current fix.

The correction considers consumption across compatible, permitted projections.
Genuine external qparams and incompatible/excluded consumers do not receive the
fusion exception. At f0ba929cb0, operator/module filters were moved before the
generic fusion fast path; round three identified this as a regression, corrected below. Registered shared-output runtime and nine partition policy
cases pass; parent reproducer passes 3/65536/2 dynamic rows. Broad validation passed: 52 Python tests, 32 INT8 C++ cases and existing INT4
GEMM. Final fork handoff is in progress. No native sources changed or rebuild needed.

Adjacent rerun: enabled static INT8 passes; dynamic INT4 has exactly the same seven baseline mismatches and values. Temporary test adaptations restored. Lint and OpenSpec pass; preparing the shared-activation correction commit.

Shared-activation delivery (historical): f0ba929cb08c15f9ee851ae716a830e34433c8b3 was pushed normally and matched the fork branch and PR 46 readback. The earlier 5bb3afb411 implementation and 7f295b497a correction are historical. This final documentation record is included before freezing the worktree. The exact final code/documentation head is recorded in task-parent logs/shared-activation-review/pr-final-readback.json and READY.md and handed to parent; no metadata commits follow that handoff. Parent owns independent Claude and runtime acceptance.


## Round-three allowlist correction and nonzero-weight investigation

Claude requested changes at e6aac4f9cc. Parent confirmed real WebGPU RMSNorm
allowlist regression. New registered RMSNorm and INT4 linear tests fail before
(two failures), then pass through both Vulkan and WebGPU partitioners with one
fused serialized delegate and llvmpipe execution. The generic _check_node_support
method now matches baseline 390bc0618f exactly; stricter operator checks apply
only to per-token activation candidates/additions. nn.Module policy is unchanged.
Shared-projection absorption and genuine external/incompatible consumer rejection
remain covered; generic fused internals are no longer asserted to obey operator
filters individually.

P3 disposition: unsupported asymmetric-weight input, not a demonstrated working
fallback regression. Identical saved per-token graphs with weight zero points 0/1
were passed through baseline Python backend files. Both serialize but abort at
Vulkan load on missing dequantize_per_channel (exit -6). Current zero=0 executes;
zero=1 gives the documented early ValueError through FusePatternsPass. CPU-only
export also fails missing out variants. Keep symmetric-weight scope and explicit
rejection; registered full-lowering coverage added. Baseline files restored exactly.
Logs: allowlist-correction/nonzero-*.log. Native sources/build unchanged; prior
32-case native result remains at the shared-activation correction.

Round-three validation milestone: 72 Python tests passed (122 deselected); five focused tests passed, including the two before/fixed allowlist regressions. Parent RMS script produces one fused delegate. Scoped lint and strict OpenSpec pass. No native files changed; previous 32 C++ INT8 cases remain prior evidence. Final correction commit and fork handoff pending.

Round-three delivery: correction 64df469e6e1acbed8cd527e0173e8d41a741a3f2 passed the commit hook, pushed normally to agent/fix-22431 and matched branch/PR 46 readback. This documentation record precedes the final frozen review head; exact final SHA is in task-parent logs/allowlist-correction/pr-final-readback.json and READY.md. No metadata commits follow handoff. Parent owns final Claude acceptance and any further findings; hardware/CI limits remain.
