# WebGPU Dawn C++ runtime 相容性修復

- 日期：2026-10-23（Asia/Taipei）
- 任務來源：[Qwen prefill PR 的 WebGPU CI](https://redirect.github.com/pytorch/executorch/pull/21873)。使用者本次明確要求處理此 CI 問題。
- Baseline：`141ecd06a09d5dfdbf738c98f413a45e37952148`
- 修復分支：`codex/webgpu-dawn-runtime-20261023`
- Fork PR base：`fix/18832-qwen35-dynamic-shape`；保留原 Qwen prefill 變更。
- 證據：[models job](https://redirect.github.com/pytorch/executorch/actions/runs/39601542015/job/119424894833)。fork 的 Issues 功能停用，因此沿用該具體 CI 問題作為任務來源。

## 驗收條件

| ID | 預期行為 | 驗證 |
| --- | --- | --- |
| AC-1 | 同一份固定版本 Dawn 可成功連結及建立 instance | baseline / fixed 原生 C++ smoke test |
| AC-2 | Conda runtime 同時用於 CMake 連結及動態載入 | 完整 setup 流程回歸測試；實際 Dawn 的 CMake build/run；ldd |
| AC-3 | 缺少、不相容的 runtime 或 Conda 安裝失敗時提早停止 | 失敗路徑測試；不下載 Dawn / SwiftShader |
| AC-4 | 保留使用者 linker flags 與既有 library path，支援空白路徑及非 Conda 環境 | 離線回歸測試 |

## 環境設定

本機為 Debian 12 x86-64、GCC 12.2、系統 libstdc++ 12；原 CI 是 Ubuntu 22.04、Clang 12.0.1、系統 libstdc++ 12.3。兩者均缺少此次 Dawn 所需符號。本機測試沒有修改系統 runtime。

測試產物在 `/workspace/executorch-artifacts/dawn-compat`，不納入 Git：

- Dawn：`v20260423.175430` / `31e25af254ab572c77054edec4946d2244e184dd`，與 CI 完全相同。
- Dawn archive SHA-256：`ac76fac090162dc1ecea5ed0f28a557bb8f49efc47faab01886105ace82b7b64`。
- 獨立測試 runtime：conda-forge `libstdcxx-14.4.0-h934c35e_7.conda`，SHA-256 `c8dce64507240af2f6132518400448080697a99c317989aff3e1ae8fc01cbdf1`。
- 生產腳本仍使用原本的 Conda `libstdcxx-ng>=14` 安裝條件，未把本機測試套件版本誤當成 CI 已安裝版本。

## 1. reproduce：原始失敗

日誌中 Ubuntu toolchain PPA 的 Packages / InRelease 回傳 503，apt 留下系統 libstdc++ 12.3。Conda 安裝 libstdcxx 16.2 已成功，但 Clang 連結器仍採用系統函式庫。`LD_LIBRARY_PATH` 單獨影響執行期，不能取代連結時的 `-L`。

以下最小程式直接使用固定 Dawn archive，不需 GPU 或 SwiftShader：

```cpp
#include <webgpu/webgpu.h>
int main() {
  WGPUInstance instance = wgpuCreateInstance(nullptr);
  if (!instance) return 1;
  wgpuInstanceRelease(instance);
  return 0;
}
```

在上述產物目錄保存為 `probe.cpp` 後：

```bash
cd /workspace/executorch-artifacts/dawn-compat
g++ probe.cpp -Idawn/include dawn/lib64/libwebgpu_dawn.a -ldl -lpthread -o probe-baseline
```

實測 exit 1，連結器報 `undefined reference`，對象是 `std::__cxx11::basic_string<...>::_M_replace_cold(...)`。原始輸出保存在 `baseline.log`。

## 2. dev：修復

1. Conda 環境不再依賴不必要的 Ubuntu PPA runtime 升級，也不吞掉 Conda 安裝失敗。
2. 新 helper 將已安裝的 `${CONDA_PREFIX}/lib` 放進 `LDFLAGS` 的 `-L` / rpath 及 `LD_LIBRARY_PATH`，保留既有設定。
3. 在下載 Dawn、建置 SwiftShader 前，使用後續建置相同的 CXX/CMake 環境，連結及執行對固定 Dawn 所需 `_M_replace_cold` 符號的檢查。它只引用函式位址，不以無效參數呼叫內部 C++ 函式。
4. 非 Conda 的系統 runtime 路徑保留；同樣必須通過實際連結/載入檢查。

CMake 僅在首次 configure 讀入 LDFLAGS；重用已配置的本機 build tree 時需重新配置乾淨的建置目錄。CI 使用乾淨 checkout。此檢查針對目前固定的 Linux x86-64 Dawn，未聲稱涵蓋所有未來 Dawn ABI 需求。

## 3. regression test

```bash
python3 .ci/scripts/tests/test_webgpu_cxx_runtime.py -v
bash -n .ci/scripts/setup-webgpu-linux-deps.sh .ci/scripts/setup-webgpu-cxx-runtime.sh
```

6 個離線測試通過，使用真實 C++ compiler/CMake 與 linker-script fixture，不下載套件、不變更 apt/Conda。

相同的兩項 setup 回歸測試亦套用到 baseline：

```bash
git show 141ecd06a09d5dfdbf738c98f413a45e37952148:.ci/scripts/setup-webgpu-linux-deps.sh > /tmp/setup-webgpu-baseline.sh
WEBGPU_TEST_SETUP_SCRIPT=/tmp/setup-webgpu-baseline.sh python3 .ci/scripts/tests/test_webgpu_cxx_runtime.py WebGpuRuntimeTest.test_full_setup_selects_runtime_before_downloading_dawn WebGpuRuntimeTest.test_conda_install_failure_is_not_ignored
```

Baseline 兩項失敗：consumer 因 `_M_replace_cold` 未定義連結失敗；Conda 安裝錯誤被吞掉，繼續嘗試下載。Fixed 兩項通過。測試攔截下載，無外部副作用。

## 解決後的 Demo

```bash
cd /workspace/executorch-artifacts/dawn-compat
g++ probe.cpp -Idawn/include dawn/lib64/libwebgpu_dawn.a \
  -L"$PWD/conda/lib" -Wl,-rpath,"$PWD/conda/lib" -ldl -lpthread -o probe-selected
./probe-selected
ldd ./probe-selected
```

實測 build/run 均 exit 0；`libstdc++.so.6` 載入獨立 Conda runtime。
另以 `source .ci/scripts/setup-webgpu-cxx-runtime.sh` 後的環境，讓 CMake 建置同一個 Dawn consumer，build/run 同樣 exit 0（`cmake-fixed.log`）。這驗證實際 helper 輸出的 flags，而非只有手動 `g++ -L` 指令有效。

## 驗收結果與限制

| ID | 狀態 | 證據 |
| --- | --- | --- |
| AC-1 | passed | 固定 Dawn：baseline exit 1；fixed build/run exit 0 |
| AC-2 | passed | 完整 setup regression、實際 Dawn CMake smoke、ldd |
| AC-3 | passed | 安裝失敗、缺少 link library、不相容 symbol 三個失敗路徑 |
| AC-4 | passed | 空白 prefix、既有 flags/path、非 Conda 測試 |

- 本機聚焦驗收通過；沒有執行完整 WebGPU models/operators 或 GPU 計算測試，也沒有在原 Ubuntu/Clang 12 runner 重跑。
- 社群 CI 狀態仍待使用者將修正整合後查驗；此 fork PR 不會自動修改既有社群 PR 分支、留言或觸發社群工作流程。
- Dawn / SwiftShader 版本與 Qwen prefill 邏輯未變更。
