#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# PR #15535 observed a 521-second maximum. The default rounds a 15% buffer to
# 600 seconds; CI jobs can calibrate it with LLAMA_EXPORT_MAX_SECONDS.
LLAMA_EXPORT_MAX_SECONDS_DEFAULT=600

llama_export_monotonic_ms() {
  "${PYTHON_EXECUTABLE:-python3}" -c \
    'import time; print(time.monotonic_ns() // 1_000_000)'
}

run_llama_export_with_timing() {
  local model="$1"
  local export_name="$2"
  shift 2

  if (($# == 0)); then
    echo "ERROR: no Llama export command was provided" >&2
    return 2
  fi

  local max_seconds="${LLAMA_EXPORT_MAX_SECONDS-${LLAMA_EXPORT_MAX_SECONDS_DEFAULT}}"
  if [[ ! "${max_seconds}" =~ ^[0-9]+$ ]] ||
    [[ ${#max_seconds} -gt 10 ]] ||
    ((10#${max_seconds} > 2147483647)); then
    echo "ERROR: LLAMA_EXPORT_MAX_SECONDS must be an integer from 0 to 2147483647; got '${max_seconds}'" >&2
    return 2
  fi

  local threshold_ms=$((10#${max_seconds} * 1000))
  local start_ms
  if ! start_ms=$(llama_export_monotonic_ms); then
    echo "ERROR: failed to read the monotonic clock before Llama export" >&2
    return 2
  fi

  local export_status
  if "$@"; then
    export_status=0
  else
    export_status=$?
  fi

  local end_ms
  if ! end_ms=$(llama_export_monotonic_ms); then
    echo "ERROR: failed to read the monotonic clock after Llama export" >&2
    if ((export_status != 0)); then
      return "${export_status}"
    fi
    return 2
  fi
  local duration_ms=$((end_ms - start_ms))

  if ((export_status != 0)); then
    echo "EXECUTORCH_LLAMA_EXPORT_METRIC model=${model} export=${export_name} duration_ms=${duration_ms} threshold_ms=${threshold_ms} result=export_failed"
    echo "Llama export command failed with status ${export_status}" >&2
    return "${export_status}"
  fi

  if ((duration_ms > threshold_ms)); then
    echo "EXECUTORCH_LLAMA_EXPORT_METRIC model=${model} export=${export_name} duration_ms=${duration_ms} threshold_ms=${threshold_ms} result=threshold_exceeded"
    echo "Llama export took ${duration_ms} ms, exceeding the ${threshold_ms} ms threshold. Review the export for a regression or calibrate LLAMA_EXPORT_MAX_SECONDS." >&2
    return 124
  fi

  echo "EXECUTORCH_LLAMA_EXPORT_METRIC model=${model} export=${export_name} duration_ms=${duration_ms} threshold_ms=${threshold_ms} result=pass"
}
