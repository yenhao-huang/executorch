/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/avg_pool_util.h>

namespace torch::executor::native {

Tensor& avg_pool1d_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    Tensor& out) {
  return avg_pool_nd_out<1>(
      ctx,
      in,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      std::nullopt,
      out);
}

} // namespace torch::executor::native
