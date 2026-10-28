# Solve Issue State

Run ID: issue-22431
Instance: et-04-22431 (existing tmux worker; no nested agents)
Started: 2026-10-28
Scope: per-token dynamic INT8 activations plus INT8 weights on Vulkan
Repository: yenhao-huang/executorch (fork writes only)
Issue: https://redirect.github.com/pytorch/executorch/issues/22431
Branch: agent/fix-22431
Baseline: 390bc0618f19276e699a255bd035a35c5296cfa8
Pull request: not created
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
| 4. Review And Commit | in_progress | final-lint-3.log, openspec-validate-final.log | Guard, layout and rank-one review findings fixed; all applicable lint clean; OpenSpec valid. Commit/publication next. |
| 5. Open And Follow Through On PR | pending | | Nothing published |
| 6. Report Result | pending | docs/howard/22431.md | Checks verified; publication outstanding |

Setup limitation: Eigen GitLab clone repeatedly fails remotely; retry or use build configuration that does not require optimized kernels. No security settings changed.

Milestone 2026-10-28 16:34 UTC: baseline token-independence test failed (36/36 mismatches, max abs 0.7333376); fixed test passed. Quantizer suite 6 passed; Vulkan passes suite 34 passed; serialized runtime 2 tests passed (16 shape/bias/output-width combinations plus repeated calls and dynamic token sizes). C++ IntegerDotAndFallback passed 12 cases; standalone CTest discovers zero tests, so the standard GTest target was executed directly. Logs: baseline-regression.log, quantizer-tests.log, passes-tests-2.log, vulkan-runtime-tests.log, cpp-runtime-tests.log.

Optional encoder measurements: blocked; required existing HF CLI environment and model root absent. download-models runtime instructions prohibit replacement without separate request. No model files downloaded.

Milestone 2026-10-28: expanded serialized Vulkan regression passed 40 shape/output-width/bias/range combinations, rank-one through rank-four, plus dynamic tokens. Shared-weight sequence passed. C++ test passed 16 cases including texture and buffer weights and dot/no-dot variants. Static tests pass baseline/fixed; enabled INT4 test has identical seven tolerance mismatches on both. Temporary decorators/API edits restored. Fork main is 449 commits behind baseline; use a new fork-only base/upstream-main-390bc0618f snapshot to keep the PR isolated.
