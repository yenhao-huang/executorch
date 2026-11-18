/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>

#include <executorch/kernels/portable/cpu/util/normalization_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <tuple>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

namespace {

template <typename CTYPE>
void layer_norm_backward(
    const Tensor& grad_out,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const std::optional<Tensor>& weight,
    executorch::aten::ArrayRef<bool> output_mask,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  using COMPUTE_T =
      typename executorch::runtime::promote_types<CTYPE, CTYPE, true>::type;

  const size_t dim = input.dim() - normalized_shape.size();
  const size_t M = getLeadingDims(input, dim);
  const size_t N = getTrailingDims(input, dim) * input.size(dim);

  const CTYPE* dy_data = grad_out.const_data_ptr<CTYPE>();
  const CTYPE* x_data = input.const_data_ptr<CTYPE>();
  const CTYPE* mean_data = mean.const_data_ptr<CTYPE>();
  const CTYPE* rstd_data = rstd.const_data_ptr<CTYPE>();
  const CTYPE* gamma_data =
      weight.has_value() ? weight.value().const_data_ptr<CTYPE>() : nullptr;

  if (output_mask[0]) {
    CTYPE* dx_data = grad_input.mutable_data_ptr<CTYPE>();
    const COMPUTE_T scale = COMPUTE_T{1} / static_cast<COMPUTE_T>(N);
    for (const auto i : c10::irange(M)) {
      const CTYPE* x = x_data + i * N;
      const CTYPE* dy = dy_data + i * N;
      CTYPE* dx = dx_data + i * N;
      const COMPUTE_T mean_value = static_cast<COMPUTE_T>(mean_data[i]);
      const COMPUTE_T a = static_cast<COMPUTE_T>(rstd_data[i]);

      COMPUTE_T ds = 0;
      COMPUTE_T db = 0;
      for (const auto j : c10::irange(N)) {
        const COMPUTE_T g =
            gamma_data ? static_cast<COMPUTE_T>(gamma_data[j]) : COMPUTE_T{1};
        const COMPUTE_T dy_g = static_cast<COMPUTE_T>(dy[j]) * g;
        ds += dy_g * static_cast<COMPUTE_T>(x[j]);
        db += dy_g;
      }
      const COMPUTE_T b = (db * mean_value - ds) * a * a * a * scale;
      const COMPUTE_T c = -b * mean_value - db * a * scale;
      for (const auto j : c10::irange(N)) {
        const COMPUTE_T g =
            gamma_data ? static_cast<COMPUTE_T>(gamma_data[j]) : COMPUTE_T{1};
        dx[j] = a * static_cast<COMPUTE_T>(dy[j]) * g +
            b * static_cast<COMPUTE_T>(x[j]) + c;
      }
    }
  }

  if (output_mask[1] || output_mask[2]) {
    CTYPE* dgamma_data =
        output_mask[1] ? grad_weight.mutable_data_ptr<CTYPE>() : nullptr;
    CTYPE* dbeta_data =
        output_mask[2] ? grad_bias.mutable_data_ptr<CTYPE>() : nullptr;
    for (const auto j : c10::irange(N)) {
      COMPUTE_T dgamma = 0;
      COMPUTE_T dbeta = 0;
      for (const auto i : c10::irange(M)) {
        const COMPUTE_T dy = static_cast<COMPUTE_T>(dy_data[i * N + j]);
        dgamma += dy *
            (static_cast<COMPUTE_T>(x_data[i * N + j]) -
             static_cast<COMPUTE_T>(mean_data[i])) *
            static_cast<COMPUTE_T>(rstd_data[i]);
        dbeta += dy;
      }
      if (dgamma_data) {
        dgamma_data[j] = dgamma;
      }
      if (dbeta_data) {
        dbeta_data[j] = dbeta;
      }
    }
  }
}

} // namespace

// native_layer_norm_backward.out(Tensor grad_out, Tensor input, SymInt[]
// normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias,
// bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2)
// -> (Tensor(a!), Tensor(b!), Tensor(c!))
std::tuple<Tensor&, Tensor&, Tensor&> native_layer_norm_backward_out(
    KernelRuntimeContext& ctx,
    const Tensor& grad_out,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    executorch::aten::ArrayRef<bool> output_mask,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(
      grad_input, grad_weight, grad_bias);

  ET_KERNEL_CHECK(
      ctx,
      check_layer_norm_backward_args(
          grad_out,
          input,
          normalized_shape,
          mean,
          rstd,
          weight,
          bias,
          output_mask,
          grad_input,
          grad_weight,
          grad_bias),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx, tensor_is_default_dim_order(input), InvalidArgument, ret_val);

  ET_KERNEL_CHECK(
      ctx,
      tensor_is_default_dim_order(mean) && tensor_is_default_dim_order(rstd),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(input, grad_out),
      InvalidArgument,
      ret_val);

  if (weight.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensor_is_default_dim_order(weight.value()),
        InvalidArgument,
        ret_val);
  }

  if (output_mask[0]) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(input, grad_input),
        InvalidArgument,
        ret_val);
    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(grad_input, input.sizes()) == Error::Ok,
        InvalidArgument,
        ret_val);
  }

  if (output_mask[1]) {
    ET_KERNEL_CHECK(
        ctx,
        tensor_is_default_dim_order(grad_weight),
        InvalidArgument,
        ret_val);
    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(grad_weight, weight.value().sizes()) == Error::Ok,
        InvalidArgument,
        ret_val);
  }

  if (output_mask[2]) {
    ET_KERNEL_CHECK(
        ctx, tensor_is_default_dim_order(grad_bias), InvalidArgument, ret_val);
    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(grad_bias, bias.value().sizes()) == Error::Ok,
        InvalidArgument,
        ret_val);
  }

  ET_SWITCH_FLOATHBF16_TYPES(
      input.scalar_type(), ctx, "native_layer_norm_backward.out", CTYPE, [&]() {
        layer_norm_backward<CTYPE>(
            grad_out,
            input,
            normalized_shape,
            mean,
            rstd,
            weight,
            output_mask,
            grad_input,
            grad_weight,
            grad_bias);
      });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
