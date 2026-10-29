# Tasks

## 1. Baseline and quantizer

- [x] 1.1 Complete standard environment/build setup and record commands and Vulkan device evidence.
- [x] 1.2 Add token-independence regression, run on baseline, implement per-token INT8 configuration, and verify fixed host graph; document compatibility option.

## 2. Fusion and reference

- [x] 2.1 Add numerically correct custom op reference and guarded graph replacement; verify custom-op versus Q/DQ and complete delegation tests across shape/bias cases.

## 3. Vulkan implementation

- [x] 3.1 Add registered runtime operation and quantize/matrix shaders with layout and resize support; compile standard Vulkan build and execute delegate tests against host reference.
- [x] 3.2 Cover constant/tiny/asymmetric inputs, batch/token/feature sizes, bias and fallback shader execution; record GPU versus software Vulkan evidence.

## 4. Integration and delivery

- [x] 4.1 Run baseline/fixed regression and relevant adjacent INT4, static and weight-only tests; investigate encoder quality/latency if prerequisites permit; update acceptance report with exact outcomes.
- [x] 4.2 Review full diff, run lint and OpenSpec validation, resolve findings, and record file changes.
- [x] 4.3 Commit and push only to fork, verify branch and separate PR URL, inspect CI/review, and update STATE/report.

## 5. Independent Claude review follow-up

- [x] 5.1 Preserve force_fp16 storage with typed shader variants and FP32 qparams; reproduce and test before/after.
- [x] 5.2 Support token counts beyond dispatch limits, including dynamic growth; retain multi-consumer coverage.
- [ ] 5.3 Run adjacent and fallback checks, lint, update evidence, push PR 46, and hand final head to parent for Claude re-review.

- [ ] 5.4 Parent runs independent Claude CLI on the final pushed revision; address further legitimate findings until accepted.
