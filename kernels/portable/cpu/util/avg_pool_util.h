/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch::executor::native {

template <size_t N>
Tensor& avg_pool_nd_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      static_cast<size_t>(in.dim()) == N + 1 ||
          static_cast<size_t>(in.dim()) == N + 2,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      out.dim() == in.dim() && tensors_have_same_dtype(in, out),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      tensor_is_default_dim_order(in) && tensor_is_default_dim_order(out),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      kernel_size.size() == 1 || kernel_size.size() == N,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      stride.empty() || stride.size() == 1 || stride.size() == N,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx, padding.size() == 1 || padding.size() == N, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx,
      !divisor_override.has_value() || *divisor_override != 0,
      InvalidArgument,
      out);

  const size_t leading = in.dim() - N;
  executorch::aten::SizesType sizes[kTensorDimensionLimit];
  for (size_t i = 0; i < leading; ++i) {
    ET_KERNEL_CHECK(
        ctx, in.size(i) > 0 || (leading == 2 && i == 0), InvalidArgument, out);
    sizes[i] = in.size(i);
  }
  std::array<int64_t, N> kernel, step, pad, input_size, output_size;
  int64_t input_volume = 1, output_volume = 1;
  for (size_t i = 0; i < N; ++i) {
    kernel[i] = kernel_size[kernel_size.size() == 1 ? 0 : i];
    step[i] = stride.empty() ? kernel[i] : stride[stride.size() == 1 ? 0 : i];
    pad[i] = padding[padding.size() == 1 ? 0 : i];
    input_size[i] = in.size(leading + i);
    ET_KERNEL_CHECK(
        ctx,
        kernel[i] > 0 && step[i] > 0 && pad[i] >= 0 &&
            pad[i] <= kernel[i] / 2 && input_size[i] > 0 &&
            kernel[i] <= std::numeric_limits<int32_t>::max() &&
            step[i] <= std::numeric_limits<int32_t>::max(),
        InvalidArgument,
        out);
    if constexpr (N == 3) {
      ET_KERNEL_CHECK(ctx, input_size[i] >= kernel[i], InvalidArgument, out);
    }
    const int64_t numerator =
        input_size[i] + 2 * pad[i] - kernel[i] + (ceil_mode ? step[i] - 1 : 0);
    output_size[i] = (numerator >= 0 ? numerator / step[i]
                                     : (numerator - step[i] + 1) / step[i]) +
        1;
    if (ceil_mode && (output_size[i] - 1) * step[i] >= input_size[i] + pad[i]) {
      --output_size[i];
    }
    ET_KERNEL_CHECK(
        ctx,
        output_size[i] > 0 &&
            output_size[i] <=
                std::numeric_limits<executorch::aten::SizesType>::max(),
        InvalidArgument,
        out);
    sizes[leading + i] = output_size[i];
    input_volume *= input_size[i];
    output_volume *= output_size[i];
  }
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {sizes, static_cast<size_t>(in.dim())}) == Error::Ok,
      InvalidArgument,
      out);

  constexpr const char* op_name = N == 1 ? "avg_pool1d.out" : "avg_pool3d.out";
  ET_SWITCH_FLOATHBF16_TYPES_AND(
      Long, in.scalar_type(), ctx, op_name, CTYPE, [&]() {
        using Acc = std::conditional_t<
            std::is_same_v<CTYPE, executorch::aten::Half> ||
                std::is_same_v<CTYPE, executorch::aten::BFloat16>,
            float,
            CTYPE>;
        const CTYPE* src = in.const_data_ptr<CTYPE>();
        CTYPE* dst = out.mutable_data_ptr<CTYPE>();
        for (int64_t index = 0; index < out.numel(); ++index) {
          int64_t position = index % output_volume;
          std::array<int64_t, N> begin, extent;
          int64_t padded_count = 1, valid_count = 1;
          for (size_t j = N; j-- > 0;) {
            const int64_t start =
                (position % output_size[j]) * step[j] - pad[j];
            position /= output_size[j];
            const int64_t end =
                std::min(start + kernel[j], input_size[j] + pad[j]);
            padded_count *= end - start;
            begin[j] = std::max<int64_t>(start, 0);
            extent[j] = std::min(end, input_size[j]) - begin[j];
            valid_count *= extent[j];
          }
          Acc sum = 0;
          for (int64_t k = 0; k < valid_count; ++k) {
            int64_t cursor = k, offset = 0, pitch = 1;
            for (size_t j = N; j-- > 0;) {
              offset += (begin[j] + cursor % extent[j]) * pitch;
              cursor /= extent[j];
              pitch *= input_size[j];
            }
            sum += static_cast<Acc>(
                src[(index / output_volume) * input_volume + offset]);
          }
          const int64_t divisor = divisor_override.value_or(
              count_include_pad ? padded_count : valid_count);
          dst[index] = static_cast<CTYPE>(sum / static_cast<Acc>(divisor));
        }
      });
  return out;
}

} // namespace torch::executor::native
