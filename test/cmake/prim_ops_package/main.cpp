/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/operator_registry.h>

#include <cstdio>

int main() {
  constexpr const char* op_names[] = {
      "aten::sym_size.int",
      "aten::sym_numel",
      "executorch_prim::sym_max.Scalar",
  };
  for (const char* op_name : op_names) {
    if (!torch::executor::hasOpsFn(op_name)) {
      std::fprintf(
          stderr, "Primitive operator is not registered: %s\n", op_name);
      return 1;
    }
  }
  std::puts("Primitive operators are registered");
  return 0;
}
