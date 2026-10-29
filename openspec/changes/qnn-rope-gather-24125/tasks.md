# Tasks

## 1. Baseline and setup

- [x] 1.1 Verify selected branch, baseline and live issue discussion; record API evidence in STATE.md.
- [x] 1.2 Install supported dependencies and QAIRT, build x86 runner with eight jobs; record exact commands, versions and exit codes.
- [x] 1.3 Run original seed-zero reproducer on HTP emulator and measure every output against eager/QDQ; record logs and metrics in the report.

## 2. Diagnosis and correction

- [x] 2.1 Trace graph shapes, quantization, layout, identities and runtime outputs; record supported or rejected hypotheses with source evidence.
- [x] 2.2 Implement the smallest demonstrated fix and focused regression; verify baseline failure and fixed success with the same test.
- [x] 2.3 Run FP16, direct-input and single-RoPE controls and original fixed model; record metrics and emulator/device limits.
- [x] 2.4 Verify physical SM8850/HTPv81 acceptance or leave the report explicitly blocked with runnable device instructions. Physical acceptance remains blocked; documentation is complete.

## 3. Review and delivery

- [x] 3.1 Review complete diff, validate OpenSpec and run relevant lint/tests; reconcile file changes and acceptance status.
- [x] 3.2 Commit and publish reviewed work only to the designated fork; verify remote branch and PR URL, checks and outstanding blockers. Fork PR 45 is open; no CI checks reported; physical acceptance blocked.

## Workflow follow-up

- Parent owns daily reporting. No upstream writes, merges, branch deletion or change archival.

## 4. Independent review revision

- [x] 4.1 Reproduce LPAI delegation regression on the reviewed head and add a registered regression.
- [x] 4.2 Pass backend selection coherently through visitors and restrict GatherNd to HTP.
- [x] 4.3 Cover repeated/out-of-order, static and activation indices/tables; verify 16a8w/16a16w HTP numerics and original RoPE correction on matched SDKs.
- [x] 4.4 Document pre-existing LPAI numerical failure and retain required workflow records.
- [x] 4.5 Commit/push the reviewed revision and verify the existing fork PR head for the parent's next Claude pass. Correction ed903011cb is verified on the existing fork branch.
- [ ] 4.6 Receive independent Claude acceptance from the parent; do not self-approve.
