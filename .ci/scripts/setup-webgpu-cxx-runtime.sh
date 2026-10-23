#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Source this after installing the runtime. CMake consumes LDFLAGS on the first
# configure; LD_LIBRARY_PATH also covers Python extensions loaded by pytest.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  if [[ ! -f "${CONDA_PREFIX}/lib/libstdc++.so" ]]; then
    echo "ERROR: conda's libstdc++.so is missing from ${CONDA_PREFIX}/lib" >&2
    return 1
  fi
  export LDFLAGS="-L\"${CONDA_PREFIX}/lib\" -Wl,-rpath,\"${CONDA_PREFIX}/lib\" ${LDFLAGS:-}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Use the same compiler and CMake linker flags as the subsequent native builds.
# Referencing (without calling) the exact libstdc++ symbol used by the pinned
# Linux Dawn archive tests link-time selection AND runtime symbol resolution.
_webgpu_check_cxx_runtime() (
  set -e
  probe_dir="$(mktemp -d)"
  trap 'rm -rf "$probe_dir"' EXIT
  cat > "$probe_dir/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.18)
project(webgpu_cxx_runtime LANGUAGES CXX)
add_executable(runtime_probe main.cpp)
CMAKE
  cat > "$probe_dir/main.cpp" <<'CPP'
extern "C" void dawn_required_symbol()
    asm("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE15_M_replace_coldEPcmPKcmm");
int main() {
  void (*volatile required)() = &dawn_required_symbol;
  return required == nullptr;
}
CPP
  cmake -S "$probe_dir" -B "$probe_dir/build" &&
    cmake --build "$probe_dir/build" &&
    "$probe_dir/build/runtime_probe" || {
      echo "ERROR: WebGPU needs a libstdc++ that can link and load Dawn's _M_replace_cold symbol. Check CXX, LDFLAGS and LD_LIBRARY_PATH." >&2
      exit 1
    }
)
_webgpu_check_cxx_runtime || return 1
unset -f _webgpu_check_cxx_runtime
