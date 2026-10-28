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
Implementation commit: 5bb3afb4119823d75e6af9a8d9d60661516226e3
Fork PR base: base/upstream-main-390bc0618f at 390bc0618f19276e699a255bd035a35c5296cfa8
Outcome: completed (fork delivery); no community submission or merge
OpenSpec: openspec/changes/vulkan-per-token-int8
Report: docs/howard/22431.md
Logs: ../logs/

Last updated: 2026-10-28

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Confirm Repository And Issue | completed | branch/HEAD checked | Exclusive worktree ownership from parent |
| 1. Check Prior Discussion For Existing Resolution | completed | issue-current.json, issue-comments.json, issue-timeline.json, pr-22432.json, pr-22372.json | Open, no comments; linked corrections do not resolve feature |
| 2. Read Issue And Repository Context | completed | required skills, AGENTS.md, CONTRIBUTING.md, OpenSpec proposal/design/spec | No community writes |
| 3. Implement And Test | completed | review-final-per-token-tests.log, final-cpp-runtime-tests.log, final-host-and-adjacent-tests.log | 5 per-token tests, 16 C++ cases, and 43 host/adjacent tests passed. |
| 4. Review And Commit | completed | all-applicable-lint-final.log, implementation-commit.log, openspec-validate-final.log | Guard/layout/rank-one findings fixed; lint, hook and OpenSpec passed. |
| 5. Open And Follow Through On PR | completed | fork-branch-readback.txt, fork-pr-readback.json, fork-pr-repositories.json | Branch and PR 46 verified on fork; 25-file diff; no checks or reviews reported. |
| 6. Report Result | completed | docs/howard/22431.md | Required acceptance passed; physical-GPU/encoder measurements remain conditional blockers. |

Setup limitation: Eigen GitLab clone repeatedly fails remotely; retry or use build configuration that does not require optimized kernels. No security settings changed.

Milestone 2026-10-28 16:34 UTC: baseline token-independence test failed (36/36 mismatches, max abs 0.7333376); fixed test passed. Quantizer suite 6 passed; Vulkan passes suite 34 passed; serialized runtime 2 tests passed (16 shape/bias/output-width combinations plus repeated calls and dynamic token sizes). C++ IntegerDotAndFallback passed 12 cases; standalone CTest discovers zero tests, so the standard GTest target was executed directly. Logs: baseline-regression.log, quantizer-tests.log, passes-tests-2.log, vulkan-runtime-tests.log, cpp-runtime-tests.log.

Optional encoder measurements: blocked; required existing HF CLI environment and model root absent. download-models runtime instructions prohibit replacement without separate request. No model files downloaded.

Milestone 2026-10-28: expanded serialized Vulkan regression passed 40 shape/output-width/bias/range combinations, rank-one through rank-four, plus dynamic tokens. Shared-weight sequence passed. C++ test passed 16 cases including texture and buffer weights and dot/no-dot variants. Static tests pass baseline/fixed; enabled INT4 test has identical seven tolerance mismatches on both. Temporary decorators/API edits restored. Fork main is 449 commits behind baseline; use a new fork-only base/upstream-main-390bc0618f snapshot to keep the PR isolated.

Delivery milestone: implementation commit 5bb3afb4119823d75e6af9a8d9d60661516226e3 verified on fork branch agent/fix-22431; fork PR 46 verified OPEN, non-draft, based on the exact requested baseline. No checks or reviews reported. Community timeline readback contains no fork cross-reference. No merge, force push, deletion, community writes or nested workers occurred. Final report and completion record are published in a documentation follow-up commit.
