#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../llama_export_timing.sh"

tests_run=0

assert_metric() {
  local output="$1"
  local threshold_ms="$2"
  local result="$3"
  local metric_pattern="^EXECUTORCH_LLAMA_EXPORT_METRIC model=stories110M export=test\.pte duration_ms=[0-9]+ threshold_ms=${threshold_ms} result=${result}$"

  if ! grep -Eq "${metric_pattern}" <<<"${output}"; then
    echo "Missing expected metric: ${metric_pattern}" >&2
    echo "Output: ${output}" >&2
    exit 1
  fi
  if [[ $(grep -c '^EXECUTORCH_LLAMA_EXPORT_METRIC ' <<<"${output}") -ne 1 ]]; then
    echo "Expected exactly one metric record: ${output}" >&2
    exit 1
  fi
}

unset LLAMA_EXPORT_MAX_SECONDS
if output=$(run_llama_export_with_timing stories110M test.pte true 2>&1); then
  status=0
else
  status=$?
fi
[[ ${status} -eq 0 ]]
assert_metric "${output}" 600000 pass
((tests_run += 1))

if output=$(LLAMA_EXPORT_MAX_SECONDS=3600 run_llama_export_with_timing stories110M test.pte true 2>&1); then
  status=0
else
  status=$?
fi
[[ ${status} -eq 0 ]]
assert_metric "${output}" 3600000 pass
((tests_run += 1))

if output=$(LLAMA_EXPORT_MAX_SECONDS=3600 run_llama_export_with_timing stories110M test.pte bash -c 'exit 23' 2>&1); then
  status=0
else
  status=$?
fi
[[ ${status} -eq 23 ]]
assert_metric "${output}" 3600000 export_failed
grep -q 'failed with status 23' <<<"${output}"
((tests_run += 1))

if output=$(LLAMA_EXPORT_MAX_SECONDS=0 run_llama_export_with_timing stories110M test.pte "${PYTHON_EXECUTABLE:-python3}" -c 'sum(range(100000))' 2>&1); then
  status=0
else
  status=$?
fi
[[ ${status} -eq 124 ]]
assert_metric "${output}" 0 threshold_exceeded
grep -q 'calibrate LLAMA_EXPORT_MAX_SECONDS' <<<"${output}"
((tests_run += 1))

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/llama-export-timing.XXXXXX")
trap 'rm -rf "${test_dir}"' EXIT
for invalid_threshold in invalid -1 ''; do
  sentinel="${test_dir}/export-ran-${tests_run}"
  if output=$(LLAMA_EXPORT_MAX_SECONDS="${invalid_threshold}" run_llama_export_with_timing stories110M test.pte touch "${sentinel}" 2>&1); then
    status=0
  else
    status=$?
  fi
  [[ ${status} -eq 2 ]]
  [[ ! -e "${sentinel}" ]]
  grep -q 'LLAMA_EXPORT_MAX_SECONDS must be an integer' <<<"${output}"
  ((tests_run += 1))
done

echo "PASS: ${tests_run} Llama export timing tests"
