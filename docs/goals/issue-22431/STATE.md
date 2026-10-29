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
Review fix commit: 7f295b497a7aad79130ce23e0124a6427b145001 (fork branch and PR head verified)
Fork PR base: base/upstream-main-390bc0618f at 390bc0618f19276e699a255bd035a35c5296cfa8
Outcome: in_progress — revision pushed and verified; parent Claude re-review pending
OpenSpec: openspec/changes/vulkan-per-token-int8
Report: docs/howard/22431.md
Logs: ../logs/

Last updated: 2026-10-29

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Confirm Repository And Issue | completed | branch/HEAD checked | Exclusive worktree ownership from parent |
| 1. Check Prior Discussion For Existing Resolution | completed | issue-current.json, issue-comments.json, issue-timeline.json, pr-22432.json, pr-22372.json | Open, no comments; linked corrections do not resolve feature |
| 2. Read Issue And Repository Context | completed | required skills, AGENTS.md, CONTRIBUTING.md, OpenSpec proposal/design/spec | No community writes |
| 3. Implement And Test | completed | claude-review/full-regression.log, claude-review/cpp-runtime-final.log | Review fixes: 50 Python tests and 32 INT8 C++ cases passed; before/fixed runtime regression 2 failures to 3 passes. |
| 4. Review And Commit | completed | claude-review/commit.log, claude-review/lint.log, claude-review/openspec-validate.log | Confirmed half/dispatch issues fixed; lint, commit hook, diff check and OpenSpec passed. |
| 5. Open And Follow Through On PR | in_progress | claude-review/fork-branch-readback.txt, claude-review/pr-code-readback.json | Revision 7f295b497a pushed normally and PR 46 head verified; parent Claude re-review pending. |
| 6. Report Result | in_progress | docs/howard/22431.md | Review evidence updated; independent acceptance pending; physical-GPU/encoder limits retained. |

Setup limitation: Eigen GitLab clone repeatedly fails remotely; retry or use build configuration that does not require optimized kernels. No security settings changed.

Milestone 2026-10-28 16:34 UTC: baseline token-independence test failed (36/36 mismatches, max abs 0.7333376); fixed test passed. Quantizer suite 6 passed; Vulkan passes suite 34 passed; serialized runtime 2 tests passed (16 shape/bias/output-width combinations plus repeated calls and dynamic token sizes). C++ IntegerDotAndFallback passed 12 cases; standalone CTest discovers zero tests, so the standard GTest target was executed directly. Logs: baseline-regression.log, quantizer-tests.log, passes-tests-2.log, vulkan-runtime-tests.log, cpp-runtime-tests.log.

Optional encoder measurements: blocked; required existing HF CLI environment and model root absent. download-models runtime instructions prohibit replacement without separate request. No model files downloaded.

Milestone 2026-10-28: expanded serialized Vulkan regression passed 40 shape/output-width/bias/range combinations, rank-one through rank-four, plus dynamic tokens. Shared-weight sequence passed. C++ test passed 16 cases including texture and buffer weights and dot/no-dot variants. Static tests pass baseline/fixed; enabled INT4 test has identical seven tolerance mismatches on both. Temporary decorators/API edits restored. Fork main is 449 commits behind baseline; use a new fork-only base/upstream-main-390bc0618f snapshot to keep the PR isolated.

Delivery milestone: implementation commit 5bb3afb4119823d75e6af9a8d9d60661516226e3 verified on fork branch agent/fix-22431; fork PR 46 verified OPEN, non-draft, based on the exact requested baseline. No checks or reviews reported. Community timeline readback contains no fork cross-reference. No merge, force push, deletion, community writes or nested workers occurred. Final report and completion record are published in a documentation follow-up commit.

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
