/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <type_traits>

#include <c10/util/irange.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using IntArrayRef = executorch::aten::ArrayRef<int64_t>;

namespace {

bool check_col2im_args(
    const Tensor& in,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_dim_order(in));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_dim_order(out));

  ET_CHECK_OR_RETURN_FALSE(
      output_size.size() == 2 && kernel_size.size() == 2 &&
          dilation.size() == 2 && padding.size() == 2 && stride.size() == 2,
      "Expected output_size, kernel_size, dilation, padding and stride to have size 2");
  ET_CHECK_OR_RETURN_FALSE(
      kernel_size[0] > 0 && kernel_size[1] > 0,
      "kernel_size should be greater than zero, but got (%" PRId64 ", %" PRId64
      ")",
      kernel_size[0],
      kernel_size[1]);
  ET_CHECK_OR_RETURN_FALSE(
      stride[0] > 0 && stride[1] > 0,
      "stride should be greater than zero, but got (%" PRId64 ", %" PRId64 ")",
      stride[0],
      stride[1]);
  ET_CHECK_OR_RETURN_FALSE(
      dilation[0] > 0 && dilation[1] > 0,
      "dilation should be greater than zero, but got (%" PRId64 ", %" PRId64
      ")",
      dilation[0],
      dilation[1]);
  ET_CHECK_OR_RETURN_FALSE(
      padding[0] >= 0 && padding[1] >= 0,
      "padding should be non-negative, but got (%" PRId64 ", %" PRId64 ")",
      padding[0],
      padding[1]);
  ET_CHECK_OR_RETURN_FALSE(
      output_size[0] > 0 && output_size[1] > 0,
      "Expected output spatial size to be positive, but got (%" PRId64
      ", %" PRId64 ")",
      output_size[0],
      output_size[1]);

  ET_CHECK_OR_RETURN_FALSE(
      (in.dim() == 2 && in.size(0) > 0 && in.size(1) > 0) ||
          (in.dim() == 3 && in.size(1) > 0 && in.size(2) > 0),
      "Expected 2D or 3D (batch mode) tensor with optional 0 dim batch size for input; in.dim() = %" ET_PRI_TENSOR_DIM,
      in.dim());

  const int64_t channels = in.size(in.dim() - 2);
  const int64_t kernel_numel = kernel_size[0] * kernel_size[1];
  ET_CHECK_OR_RETURN_FALSE(
      channels % kernel_numel == 0,
      "Expected size of input's channel dimension (%" PRId64
      ") to be divisible by the product of kernel_size (%" PRId64 ")",
      channels,
      kernel_numel);

  int64_t num_blocks = 1;
  for (const auto i : c10::irange(2)) {
    const int64_t span = output_size[i] + 2 * padding[i] -
        dilation[i] * (kernel_size[i] - 1) - 1;
    ET_CHECK_OR_RETURN_FALSE(
        span >= 0,
        "Calculated number of sliding blocks along dimension %d must be at least one",
        static_cast<int>(i));
    num_blocks *= span / stride[i] + 1;
  }
  ET_CHECK_OR_RETURN_FALSE(
      in.size(in.dim() - 1) == num_blocks,
      "Expected size of input's last dimension (%" PRId64
      ") to match the calculated number of sliding blocks (%" PRId64 ")",
      static_cast<int64_t>(in.size(in.dim() - 1)),
      num_blocks);

  return true;
}

} // namespace

Tensor& col2im_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_col2im_args(
          in, output_size, kernel_size, dilation, padding, stride, out),
      InvalidArgument,
      out);

  const bool batched = in.dim() == 3;
  const int64_t batch = batched ? in.size(0) : 1;
  const int64_t kernel_h = kernel_size[0];
  const int64_t kernel_w = kernel_size[1];
  const int64_t channels = in.size(in.dim() - 2) / (kernel_h * kernel_w);
  const int64_t out_h = output_size[0];
  const int64_t out_w = output_size[1];
  const int64_t blocks_h =
      (out_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) / stride[0] +
      1;
  const int64_t blocks_w =
      (out_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) / stride[1] +
      1;

  executorch::aten::SizesType out_sizes[4];
  size_t out_ndim = 0;
  if (batched) {
    out_sizes[out_ndim++] = batch;
  }
  out_sizes[out_ndim++] = channels;
  out_sizes[out_ndim++] = out_h;
  out_sizes[out_ndim++] = out_w;

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {out_sizes, out_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "col2im.out";

  ET_SWITCH_FLOATHBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&]() {
    using ACC = std::conditional_t<
        std::is_same_v<CTYPE, executorch::aten::Half> ||
            std::is_same_v<CTYPE, executorch::aten::BFloat16>,
        float,
        CTYPE>;
    const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();
    const int64_t num_blocks = blocks_h * blocks_w;

    for (const auto plane : c10::irange(batch * channels)) {
      const CTYPE* const in_plane =
          in_data + plane * kernel_h * kernel_w * num_blocks;
      CTYPE* const out_plane = out_data + plane * out_h * out_w;
      for (const auto h : c10::irange(out_h)) {
        for (const auto w : c10::irange(out_w)) {
          ACC accum = 0;
          for (const auto kh : c10::irange(kernel_h)) {
            const int64_t h_offset = h + padding[0] - kh * dilation[0];
            const int64_t bh = h_offset / stride[0];
            if (h_offset < 0 || h_offset % stride[0] != 0 || bh >= blocks_h) {
              continue;
            }
            for (const auto kw : c10::irange(kernel_w)) {
              const int64_t w_offset = w + padding[1] - kw * dilation[1];
              const int64_t bw = w_offset / stride[1];
              if (w_offset < 0 || w_offset % stride[1] != 0 || bw >= blocks_w) {
                continue;
              }
              accum += static_cast<ACC>(
                  in_plane
                      [(kh * kernel_w + kw) * num_blocks + bh * blocks_w + bw]);
            }
          }
          out_plane[h * out_w + w] = static_cast<CTYPE>(accum);
        }
      }
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
