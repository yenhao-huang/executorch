/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <type_traits>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using IntArrayRef = executorch::aten::ArrayRef<int64_t>;

Tensor& adaptive_avg_pool1d_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef output_size,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_adaptive_avg_pool1d_args(in, output_size, out),
      InvalidArgument,
      out);

  size_t output_ndim = 0;
  executorch::aten::SizesType output_sizes[kTensorDimensionLimit];
  get_adaptive_avg_pool1d_out_target_size(
      in, output_size, output_sizes, &output_ndim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType in_type = in.scalar_type();

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "adaptive_avg_pool1d.out";

  ET_SWITCH_FLOATHBF16_TYPES(in_type, ctx, op_name, CTYPE, [&]() {
    using ACC = std::conditional_t<
        std::is_same_v<CTYPE, executorch::aten::Half> ||
            std::is_same_v<CTYPE, executorch::aten::BFloat16>,
        float,
        CTYPE>;

    const CTYPE* const in_ptr = in.const_data_ptr<CTYPE>();
    CTYPE* const out_ptr = out.mutable_data_ptr<CTYPE>();

    const int64_t in_L = in.size(in.dim() - 1);
    const int64_t out_L = out.size(out.dim() - 1);
    const int64_t rows = in.numel() / in_L;

    for (int64_t r = 0; r < rows; ++r) {
      const CTYPE* row_in = in_ptr + r * in_L;
      CTYPE* row_out = out_ptr + r * out_L;
      for (int64_t ol = 0; ol < out_L; ++ol) {
        const int64_t start = ol * in_L / out_L;
        const int64_t end = ((ol + 1) * in_L + out_L - 1) / out_L;
        ACC sum = 0;
        for (int64_t il = start; il < end; ++il) {
          sum += static_cast<ACC>(row_in[il]);
        }
        row_out[ol] = static_cast<CTYPE>(sum / static_cast<ACC>(end - start));
      }
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
