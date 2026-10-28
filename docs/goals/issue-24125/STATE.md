# Solve Issue State

Run ID: issue-24125
Instance: et-03-24125 (existing tmux worker; no nested workers)
Started: 2026-10-28
Scope: QNN multi-RoPE gathered tensors only
Repository: yenhao-huang/executorch (writes); pytorch/executorch (reads)
Issue: [ExecuTorch 24125](https://redirect.github.com/pytorch/executorch/issues/24125)
Branch: agent/fix-24125
Baseline: 390bc0618f19276e699a255bd035a35c5296cfa8
Pull request: none
Report: docs/howard/24125.md
OpenSpec: openspec/changes/qnn-rope-gather-24125

Last updated: 2026-10-28

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Confirm Repository And Issue | completed | git status, rev-parse HEAD, worktree list, remote -v | Clean requested branch at baseline |
| 1. Check Prior Discussion For Existing Resolution | completed | ../logs/{issue-current,comments,timeline,linked-search,commit-search}.json | Open; zero comments/events; no linked fix found |
| 2. Read Issue And Repository Context | completed | AGENTS.md, CONTRIBUTING.md, required skills, parent-notes.md | No physical device or existing SDK supplied |
| 3. Implement And Test | completed (emulator) | ../logs/regression-baseline-final.log, regression-fixed.log, regression-fixed-2.47.log | Both SDKs reproduce; equivalent GatherNd fixes original model; physical acceptance blocked |
| 4. Review And Commit | in_progress | ../logs/lint-all-files.log, buck-parity-ready.log; full diff review | Minimal builder fix; scoped lint, strict OpenSpec and Buck pass |
| 5. Open And Follow Through On PR | pending | | Fork only; no upstream references except redirect host |
| 6. Report Result | in_progress | docs/howard/24125.md | Physical SM8850 acceptance blocked |

## Environment and commands

Existing isolated worktree verified; no worktree created. Python .venv is seeded
3.12.16. Debian 12 x86_64, no adb found. Workspace has 13 GB available;
use /tmp for SDK and pip cache. Builds capped at 8 jobs.

Started repository-supported setup:

```bash
git submodule update --init --recursive --jobs 8 -- extension/llm/tokenizers third-party/ao third-party/flatbuffers third-party/flatcc third-party/gflags third-party/json third-party/pybind11 third-party/pocketfft third-party/prelude shim kernels/optimized/third-party/eigen backends/xnnpack/third-party
source .venv/bin/activate
EXECUTORCH_BUILD_CUDA=OFF MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 PIP_CACHE_DIR=/tmp/et-24125-pip-cache ./install_requirements.sh
python3 backends/qualcomm/scripts/download_qnn_sdk.py --dst-folder /tmp/et-24125-sdk-2.51.0.260929
```

The initial SDK command exited 1 (system Python lacks requests); retry with
`.venv/bin/python` succeeded. SDK installed at /tmp/et-24125-sdk-2.51.0.260929.
No account/license gate encountered. Repository helper `--install-sdk` staged
LLVM 14 libc++ under /tmp/et-24125-cache/executorch/qnn/libcxx-14.0.0.
Main pins 2.51.0.260929; reporter used 2.47.0.260601 and a v1.4.1 fork. Linux x64 HTP
emulation is documented in docs/source/backends-qualcomm.md. It does not prove
physical HTPv81 correctness. No licenses accepted, no security settings changed.

## Setup milestone

- Core dependencies installed: torch 2.14.0+cpu, torchao 0.19.0.dev20260907+cpu,
  numpy 2.5.4. `install_requirements.sh` exited 1 at missing tokenizer checkout;
  required submodules subsequently initialized (submodules-retry.log).
- Eigen clone failed twice because GitLab reports server load; host QNN build
  does not require optimized Eigen kernels and has proceeded without it.
- Initial editable install failed in unrelated CMSIS-NN pybind setup (missing
  pybind11 Python dependency); use the supported minimal AOT-only install.
- Running `QNN_SDK_ROOT=/tmp/et-24125-sdk-2.51.0.260929
  ./backends/qualcomm/scripts/build.sh --skip_linux_android --job_number 8`;
  build log ../logs/build-x86.log. No backend edits.
- Downloading reporter SDK from the same official repository-configured route,
  substituting 2.47.0.260601; ../logs/sdk-download-2.47.log.
- Exact issue Python code extracted to ../logs/reproduce_original.py;
  SHA256 3b03ee140e97e3c97d67bb7606779d545aee98ef7d182d41e6ea04eae06a46b3.
- Source paths inspected: builders/op_index.py, builders/node_visitor.py,
  _passes/{annotate_quant_attrs,insert_requantize,insert_io_qdq,layout_transform}.py,
  aot/wrappers/TensorWrapper.cpp, runtime/{QnnExecuTorchBackend,QnnManager}.cpp.
  Shared wrapper caching and native/output tensor types are expected; no proven
  lifetime defect. Runtime binds outputs to caller buffers before GraphExecute.

## Baseline reproduced on HTP emulator

Host build exited 0. Minimal editable AOT install exited 0; added the declared
pandas dependency for Qualcomm devtools imports. Exact original export exited 0
(`../logs/baseline-2.51/export-original-retry.log`). SDK build ID
v2.51.0.260929181845. Original runner command exited 0:

```bash
build-x86/examples/qualcomm/executor_runner/qnn_executor_runner \
  --model_path ../logs/baseline-2.51/rope_gather.pte \
  --input_list_path ../logs/baseline-2.51/input_list.txt \
  --output_folder_path ../logs/baseline-2.51/outputs \
  --etdump_path ../logs/baseline-2.51/etdump.etdp
```

LD_LIBRARY_PATH uses build-x86/lib, SDK lib/x86_64-linux-clang, and the extracted
LLVM bundle's lib/x86_64-unknown-linux-gnu (contains the required SONAME links).
Output 2 relative L2 **0.7683357**, max absolute error 24.418911; host QDQ relative
L2 **0.0016637330**. This reproduces the numerical defect on unchanged baseline.
This records the baseline milestone; the subsequent fix is described below. Logs at
../logs/baseline-2.51/execute.log; raw outputs and expected_out2.npy alongside.

## Diagnosis and implementation milestone

Matched QAIRT 2.47 build ID v2.47.0.260601114230 reproduces output-2 relative L2
0.7683357 too. Standalone SDK qnn-net-run produces all three outputs bitwise
identical to ExecuTorch (baseline-2.51/sdk-boundary.json). Consistent Gather shapes,
quantization and wrapper identities were inspected. No lifetime defect is proven.
Read-only wrapper capture does not add debug outputs; APP_READ tensor dumping
changes optimization and hides the defect, so is not acceptance evidence.

Implemented op_index.py workaround for axis-zero, rank-two UFIXED_POINT16 table
and vector indices: reshape indices [N] to [N,1], then GatherNd(batch_dims=0).
No changed quantization, copied tables or extra graph outputs. All other forms
retain Gather. Added SharedGatherRoPE model and registered HTP regression.

Same test before production edit: exit 1, one numerical failure on original k1;
after edit: exit 0, both (6,320,128) and (2,65,64) shapes pass. Both SDK builds
pass the fixed test. Existing FP16/quantized index tests pass on axes 0/1/2.
The final tolerances are atol=.1, rtol=.01 against QDQ; an earlier .02 absolute
tolerance tripped ordinary BMM rounding and is not the claimed baseline failure.

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/.." QNN_SDK_ROOT=/tmp/et-24125-sdk-2.51.0.260929
export LD_LIBRARY_PATH="$PWD/build-x86/lib:$QNN_SDK_ROOT/lib/x86_64-linux-clang" OMP_NUM_THREADS=8
python backends/qualcomm/tests/test_qnn_delegate.py TestQNNQuantizedOperator.test_qnn_backend_shared_gather_rope --soc_model SM8850 --build_folder build-x86 --enable_x86_64
python backends/qualcomm/tests/test_qnn_delegate.py TestQNNQuantizedOperator.test_qnn_backend_index TestQNNFloatingPointOperator.test_qnn_backend_index --soc_model SM8850 --build_folder build-x86 --enable_x86_64
```

Fixed original model k1 relative L2: 2.51 = 0.001663680654; 2.47 = 0.001663675997.
All original exports and runners exit 0. Exact unmodified reporter script rerun
in fixed-original-2.51 matches every captured fixed output bitwise. FP16/direct
controls remain accurate; single output-only RoPE and multiply-only models also
fail on baseline and pass after correction. Metrics: ../logs/numerical-results.json.
Full report: docs/howard/24125.md. AC-1/2/3 passed on emulator, AC-4 blocked.

## Matching older SDK build commands (all exited 0)

Official SDK extracted at /tmp/et-24125-qairt-2.47/qairt/2.47.0.260601.

```bash
cmake -S . -B /tmp/et-24125-build-2.47 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=/tmp/et-24125-build-2.47 -DQNN_SDK_ROOT=/tmp/et-24125-qairt-2.47/qairt/2.47.0.260601 -DEXECUTORCH_BUILD_QNN=ON -DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_BUILD_EXTENSION_LLM=ON -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON -DEXECUTORCH_ENABLE_LOGGING=ON -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build /tmp/et-24125-build-2.47 -j 8 --target install
cmake -S examples/qualcomm -B /tmp/et-24125-build-2.47/examples/qualcomm -DCMAKE_BUILD_TYPE=RelWithDebInfo '-DCMAKE_PREFIX_PATH=/tmp/et-24125-build-2.47;/tmp/et-24125-build-2.47/third-party/gflags' -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" -DSUPPORT_REGEX_LOOKAHEAD=ON -DBUILD_TESTING=OFF -DEXECUTORCH_ENABLE_LOGGING=ON
cmake --build /tmp/et-24125-build-2.47/examples/qualcomm -j 8
```

For 2.47 runs, copied that build's PyQnnManagerAdaptor into the generated local
backend Python directory and set QNN_SDK_ROOT/LD_LIBRARY_PATH/build_folder to its
SDK/build. After runs, restored the 2.51 adaptor, SHA256
04e761a67b93ad25bc755d58b582ccbde9b05e418faeabf9b6fd8d28fef6e425.
No adaptor binaries are committed.

## Review and remaining checks

`lintrunner init` initially could not create ~/.local/share (OS permission error).
Task-local XDG_DATA_HOME=/tmp/et-24125-lint-data allowed normal initialization,
exit 0. Scoped lint exits 0 after binding loop variables in the sample helper:

```bash
XDG_DATA_HOME=/tmp/et-24125-lint-data lintrunner backends/qualcomm/builders/op_index.py backends/qualcomm/tests/models.py backends/qualcomm/tests/test_qnn_delegate.py
```

Pinned Buck 2025-05-06 downloaded locally. Initialized googletest, configured the
SDK using command-line qualcomm.qnn_sdk_root, and used extracted LLVM clang 14.
Initial builds reached Python.h discovery failure in third-party pybind11/pytree
and the QNN adaptor. A symlinked python3-config incorrectly inferred /tmp as its
installation prefix. Replaced that task-local symlink with a launcher invoking
the original seeded python3.12-config; no shared tool installation was modified.
Fresh task-isolated Buck build now exits 0 (buck-parity-ready.log, 291 commands):

```bash
export PATH="/tmp/et-24125-tools:/tmp/et-24125-cache/executorch/qnn/libcxx-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/bin:$PATH"
QNN_SDK_ROOT=/tmp/et-24125-sdk-2.51.0.260929 /tmp/et-24125-buck2 --isolation-dir issue24125ready build -j 8 -c qualcomm.qnn_sdk_root=/tmp/et-24125-sdk-2.51.0.260929 //backends/qualcomm/...
```

No unrelated C++/build source changes or automatic fixes made. Required CMake
parity recheck with --skip_linux_android --no_clean --job_number 8 exits 0
(cmake-parity-recheck.log).
Final registered suite (regression plus both adjacent index tests) exits 0,
three tests. Scoped all-file lint exits 0; OpenSpec strict validation exits 0.

Fork main is 449 commits behind the required baseline. Delivery will use a new
fork base branch base/upstream-main-390bc0618f at the exact baseline, so the PR
contains only this issue. No default branch updates or rebases are planned.
Only the feature/base branches and fork PR are authorized destinations. New
outgoing text must use redirect.github.com for all community references.
