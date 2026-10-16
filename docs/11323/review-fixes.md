# Embedding bag review fixes

Source: [review](https://redirect.github.com/pytorch/executorch/pull/23871#issuecomment-6194265247).
Baseline: 4d616fec1b2b3cf9aeb02eff0782e668f077597b. Target: yenhao-huang/executorch only.

## Acceptance criteria

| ID | Expected behavior | Verification |
| --- | --- | --- |
| AC1 | Reject negative/out-of-range indices before padding or weight reads | Regression failure tests, including zero-width outputs |
| AC2 | Reject nonzero first, decreasing and out-of-range offsets | Failure tests; preserve valid repeated/partial final offsets |
| AC3 | Accumulate Half/BFloat16 sum/mean in float | Large-bag precision regression across supported dtypes |
| AC4 | Match ATen promoted Int/Long auxiliary outputs; validate optional weights/output types | ATen probes and typed tests |
| AC5 | Existing supported modes remain correct; changed files lint clean | Focused and adjacent tests; formatting |

## Environment and reproduction

Debian Linux; /data/venv_et_batch2026 Python with torch 2.14.0+cpu.
ATen accepts mixed Int/Long index tensors and promotes auxiliary outputs;
both Int inputs produce Int auxiliary outputs. Thus the bot's tentative
same-dtype recommendation is not adopted. ATen accepts a final offset smaller
than indices length with include_last_offset=true, so do not require equality.

Implementation and baseline/fixed evidence are recorded below; local checks passed.

## Root cause and fix

Index values previously reached weight pointer arithmetic unchecked; -1 was
silently skipped as default padding. Validate all indices and offsets before
resizing/writing outputs, including zero-width output cases. Reject invalid
per-sample-weight rank/length even when the optional tensor is empty.

Use the promoted Int/Long type for auxiliary tensors. Accumulate each feature
in float for Half/BFloat16 (keep Float/Double native), then divide/cast once.
Feature-wise accumulation avoids adding scratch-memory requirements; performance
was not benchmarked. Preserve empty bags, padding, mixed index types and valid
partial final offsets. Apply clang-format 18.1.3 to both C++ files.

## Regression evidence and demo

The kernel and test translation units were freshly compiled against the PR's
headers and existing portable runtime/gtest libraries from the earlier batch
CMake build. This is a focused relink, not a clean rebuild of the whole runtime.
The committed helper makes that exact validation reproducible:

```bash
python3 docs/11323/verify_review.py \
  --runtime-build /workspace/executorch-worktrees/batch-2026-10/executorch/cmake-out \
  --out /tmp/embedding-review-build
/tmp/embedding-review-build/test

git show 4d616fec1b:kernels/portable/cpu/op_embedding_bag.cpp > /tmp/embedding-baseline.cpp
python3 docs/11323/verify_review.py \
  --runtime-build /workspace/executorch-worktrees/batch-2026-10/executorch/cmake-out \
  --kernel-source /tmp/embedding-baseline.cpp --out /tmp/embedding-baseline-build
/tmp/embedding-baseline-build/test --gtest_filter='*NegativeIndexWithoutPaddingRejected:*NonzeroFirstOffsetRejected:*LargeHalfBagPrecision:*LargeBFloat16BagPrecision'

python3 docs/11323/verify_review.py \
  --runtime-build /workspace/executorch-worktrees/batch-2026-10/executorch/cmake-out \
  --sanitize --out /tmp/embedding-asan-build
/tmp/embedding-asan-build/test
```

| Scenario | Baseline | Fixed |
| --- | --- | --- |
| Negative -1 without padding | Incorrect success; failure regression fails | InvalidArgument, passes |
| Nonzero first offset | Incorrect success; failure regression fails | InvalidArgument, passes |
| 4096 Half values of 1 | Sum 2048 instead of 4096; mean also wrong | Sum 4096, mean 1 |
| 4096 BFloat16 values of 1 | Sum 256 instead of 4096; mean also wrong | Sum 4096, mean 1 |
| All embedding-bag cases | Four safe selected regression cases fail (exit 1) | 27 tests pass (exit 0) |
| ASan + UBSan on changed translation units | Not run on unsafe baseline reads | 27 tests pass, no diagnostics |
| Adjacent embedding/sum/mean/max tests | Existing cached test objects reused | 104 pass, 6 existing conditional skips; 3 disabled tests |

The adjacent run relinked the cached portable test objects with the new kernel
and test object, filtering `*Embedding*:*Sum*:*Mean*:*Max*`. Runtime libraries
were not sanitizer-instrumented. No hardware/backend validation is claimed.

## CI diagnosis

The upstream lint job reports CLANGFORMAT on the kernel and test files; the
matching formatter's dry-run now passes. The NXP job failed while cloning Eigen:
`GitLab is currently unable to handle this request due to load`, before kernel
execution. This is an external dependency fetch failure, not evidence of a
kernel test failure. It needs a user/maintainer rerun; no upstream job was rerun.

## Acceptance

AC1–AC5 passed focused local validation. Remaining integration checks: clean CI
build and the NXP dependency fetch rerun. Upstream PR remains untouched; these
fixes are delivered on a separate personal-fork branch for user review/transfer.
Authored with OpenAI Codex.
