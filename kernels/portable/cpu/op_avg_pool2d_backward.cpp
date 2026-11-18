/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>

#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using IntArrayRef = executorch::aten::ArrayRef<int64_t>;

namespace {

bool check_avg_pool2d_backward_args(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    const Tensor& grad_input) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(grad_output, input));

  ET_CHECK_OR_RETURN_FALSE(
      check_avg_pool2d_args(
          input,
          kernel_size,
          stride,
          padding,
          ceil_mode,
          count_include_pad,
          divisor_override,
          grad_input),
      "Invalid avg_pool2d arguments");

  ET_LOG_AND_RETURN_IF_FALSE(
      stride_is_valid(stride, /*kernel_ndim=*/2, /*allow_empty=*/true));

  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_dim_order(grad_output));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_dim_order(input));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_dim_order(grad_input));

  // Guarantees a non-negative numerator in _kernel_output_size_helper, whose
  // truncating division would otherwise round up instead of down.
  for (const auto i : c10::irange(2)) {
    const int64_t k = val_at(kernel_size, i);
    const int64_t s = val_at(stride, i, /*default_value=*/k);
    const int64_t padded =
        input.size(input.dim() - 2 + i) + 2 * val_at(padding, i);
    ET_CHECK_OR_RETURN_FALSE(
        padded - k + (ceil_mode ? s - 1 : 0) >= 0,
        "Pooling parameters produce a non-positive output size");
  }

  size_t output_ndim = 0;
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  executorch::aten::SizesType output_sizes[kTensorDimensionLimit];
  get_avg_pool2d_out_target_size(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      output_sizes,
      &output_ndim);

  ET_LOG_AND_RETURN_IF_FALSE(
      output_size_is_valid({output_sizes, output_ndim}, 2));

  ET_CHECK_OR_RETURN_FALSE(
      grad_output.dim() == input.dim(),
      "grad_output should have same number of dimensions as input; grad_output.dim() = %" ET_PRI_TENSOR_DIM
      ", input.dim() = %" ET_PRI_TENSOR_DIM,
      grad_output.dim(),
      input.dim());

  ET_LOG_AND_RETURN_IF_FALSE(
      tensor_has_expected_size(grad_output, {output_sizes, output_ndim}));

  return true;
}

int64_t window_extent(
    int64_t out_idx,
    int64_t in_size,
    int64_t k,
    int64_t s,
    int64_t p,
    bool count_include_pad) {
  const int64_t start = out_idx * s - p;
  const int64_t end = std::min(start + k, in_size + p);
  if (count_include_pad) {
    return end - start;
  }
  return std::min(end, in_size) - std::max(start, static_cast<int64_t>(0));
}

// Each grad_input element gathers grad_output / divisor from every window
// that covers it, so accumulation happens in ACC and no zero-fill is needed.
template <typename CTYPE, typename ACC>
void avg_pool2d_backward_impl(
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    Tensor& grad_input) {
  const CTYPE* const grad_output_data = grad_output.const_data_ptr<CTYPE>();
  CTYPE* const grad_input_data = grad_input.mutable_data_ptr<CTYPE>();

  const int64_t ndim = grad_input.dim();
  const int64_t planes =
      ndim == 3 ? grad_input.size(0) : grad_input.size(0) * grad_input.size(1);
  const int64_t in_H = grad_input.size(ndim - 2);
  const int64_t in_W = grad_input.size(ndim - 1);
  const int64_t out_H = grad_output.size(ndim - 2);
  const int64_t out_W = grad_output.size(ndim - 1);

  const int64_t k_H = val_at(kernel_size, 0);
  const int64_t k_W = val_at(kernel_size, 1);
  const int64_t s_H = val_at(stride, 0, /*default_value=*/k_H);
  const int64_t s_W = val_at(stride, 1, /*default_value=*/k_W);
  const int64_t p_H = val_at(padding, 0);
  const int64_t p_W = val_at(padding, 1);

  for (const auto plane : c10::irange(planes)) {
    const CTYPE* const grad_out_plane =
        grad_output_data + plane * out_H * out_W;
    CTYPE* const grad_in_plane = grad_input_data + plane * in_H * in_W;

    for (const auto ih : c10::irange(in_H)) {
      const int64_t h_num = ih + p_H - k_H + 1;
      const int64_t oh0 = h_num <= 0 ? 0 : (h_num + s_H - 1) / s_H;
      const int64_t oh1 = std::min(out_H, (ih + p_H) / s_H + 1);

      for (const auto iw : c10::irange(in_W)) {
        const int64_t w_num = iw + p_W - k_W + 1;
        const int64_t ow0 = w_num <= 0 ? 0 : (w_num + s_W - 1) / s_W;
        const int64_t ow1 = std::min(out_W, (iw + p_W) / s_W + 1);

        ACC accum = 0;
        for (int64_t oh = oh0; oh < oh1; ++oh) {
          const int64_t extent_h =
              window_extent(oh, in_H, k_H, s_H, p_H, count_include_pad);
          for (int64_t ow = ow0; ow < ow1; ++ow) {
            const int64_t divisor = divisor_override.has_value()
                ? divisor_override.value()
                : extent_h *
                    window_extent(ow, in_W, k_W, s_W, p_W, count_include_pad);
            accum += static_cast<ACC>(grad_out_plane[oh * out_W + ow]) /
                static_cast<ACC>(divisor);
          }
        }
        grad_in_plane[ih * in_W + iw] = static_cast<CTYPE>(accum);
      }
    }
  }
}

} // namespace

Tensor& avg_pool2d_backward_out(
    KernelRuntimeContext& ctx,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    Tensor& grad_input) {
  ET_KERNEL_CHECK(
      ctx,
      check_avg_pool2d_backward_args(
          grad_output,
          input,
          kernel_size,
          stride,
          padding,
          ceil_mode,
          count_include_pad,
          divisor_override,
          grad_input),
      InvalidArgument,
      grad_input);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(grad_input, input.sizes()) == Error::Ok,
      InvalidArgument,
      grad_input);

  static constexpr auto name = "avg_pool2d_backward.grad_input";

  ET_SWITCH_FLOATHBF16_TYPES_AND(
      Long, input.scalar_type(), ctx, name, CTYPE, [&]() {
        using ACC = typename executorch::runtime::
            promote_types<CTYPE, CTYPE, /*half_to_float=*/true>::type;
        avg_pool2d_backward_impl<CTYPE, ACC>(
            grad_output,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            divisor_override,
            grad_input);
      });

  return grad_input;
}

} // namespace native
} // namespace executor
} // namespace torch
