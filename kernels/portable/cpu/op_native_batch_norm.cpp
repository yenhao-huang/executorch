/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstring>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/dtype_util.h>
#include <executorch/kernels/portable/cpu/util/normalization_ops_util.h>
#include <executorch/kernels/portable/cpu/vec_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using SizesType = executorch::aten::SizesType;

namespace {

// Reads a batch-norm parameter element-by-element, converting from whatever
// float dtype the tensor carries. A parameter is allowed to be float32 while
// the input is float16/bfloat16, so its dtype cannot be assumed to be CTYPE.
template <typename CTYPE_COMPUTE, ET_OPERATOR_NAME_TYPE op_name>
class ParamReader {
 public:
  ParamReader(KernelRuntimeContext& ctx, const Tensor& t)
      : bytes_(reinterpret_cast<const char*>(t.const_data_ptr())),
        element_size_(t.element_size()),
        load_(utils::internal::get_load_to_compute_fn_floathbf16<
              CTYPE_COMPUTE,
              op_name>(ctx, t)) {}

  ParamReader(KernelRuntimeContext& ctx, const std::optional<Tensor>& t) {
    if (t.has_value()) {
      bytes_ = reinterpret_cast<const char*>(t.value().const_data_ptr());
      element_size_ = t.value().element_size();
      load_ = utils::internal::
          get_load_to_compute_fn_floathbf16<CTYPE_COMPUTE, op_name>(
              ctx, t.value());
    }
  }

  // False only when the dtype was rejected, which already failed the context.
  bool ok() const {
    return bytes_ == nullptr || load_ != nullptr;
  }

  CTYPE_COMPUTE at(size_t i) const {
    return load_(&bytes_[i * element_size_]);
  }

  CTYPE_COMPUTE at_or(size_t i, CTYPE_COMPUTE fallback) const {
    return bytes_ == nullptr ? fallback : at(i);
  }

 private:
  const char* bytes_ = nullptr;
  size_t element_size_ = 0;
  utils::internal::load_to_compute_fn<CTYPE_COMPUTE> load_ = nullptr;
};

} // namespace

std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_no_training_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& invstd_out) {
  (void)ctx;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, invstd_out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(mean_out, {0}) == Error::Ok, InvalidArgument, ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(invstd_out, {0}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      check_batch_norm_args(
          in,
          weight,
          bias,
          running_mean,
          running_var,
          momentum,
          eps,
          out,
          mean_out,
          invstd_out),
      InvalidArgument,
      ret_val);

  // For now, only support the contiguous dim order
  ET_KERNEL_CHECK(
      ctx,
      is_contiguous_dim_order(in.dim_order().data(), in.dim_order().size()),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(in, out, mean_out, invstd_out),
      InvalidArgument,
      ret_val);

  if (weight.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(in, weight.value()),
        InvalidArgument,
        ret_val);
  }

  if (bias.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(in, bias.value()),
        InvalidArgument,
        ret_val);
  }

  size_t C_dim = in.dim() >= 1 ? 1 : 0;
  size_t C = in.size(C_dim);
  size_t outer = getLeadingDims(in, C_dim);
  size_t inner = getTrailingDims(in, C_dim);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  ET_DEFINE_OPERATOR_NAME(name, "native_batch_norm_legit_no_training.out");

  ET_SWITCH_FLOATHBF16_TYPES(in.scalar_type(), ctx, name, CTYPE, [&] {
    // Parameters may be float32 for a half/bfloat16 input, and torch reduces in
    // float either way, so accumulate in the promoted type.
    using COMPUTE_T =
        typename executorch::runtime::promote_types<CTYPE, CTYPE, true>::type;

    const CTYPE* in_data = in.const_data_ptr<CTYPE>();
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

    const auto mean_param = ParamReader<COMPUTE_T, name>(ctx, running_mean);
    const auto var_param = ParamReader<COMPUTE_T, name>(ctx, running_var);
    const auto weight_param = ParamReader<COMPUTE_T, name>(ctx, weight);
    const auto bias_param = ParamReader<COMPUTE_T, name>(ctx, bias);
    if (!mean_param.ok() || !var_param.ok() || !weight_param.ok() ||
        !bias_param.ok()) {
      return;
    }

    const COMPUTE_T eps_ct = static_cast<COMPUTE_T>(eps);
    for (size_t i = 0; i < outer; ++i) {
      for (size_t c = 0; c < C; ++c) {
        const COMPUTE_T mean = mean_param.at(c);
        const COMPUTE_T var = var_param.at(c);
        const COMPUTE_T invstd = COMPUTE_T(1) / std::sqrt(var + eps_ct);
        const COMPUTE_T weight_val = weight_param.at_or(c, COMPUTE_T(1));
        const COMPUTE_T bias_val = bias_param.at_or(c, COMPUTE_T(0));
        for (size_t j = 0; j < inner; ++j) {
          *out_data = static_cast<CTYPE>(
              (static_cast<COMPUTE_T>(*in_data) - mean) * invstd * weight_val +
              bias_val);
          out_data++;
          in_data++;
        }
      }
    }
  });

  return ret_val;
}

std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    Tensor& running_mean,
    Tensor& running_var,
    bool training,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& invstd_out) {
  (void)ctx;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, invstd_out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      training == false,
      InvalidArgument,
      ret_val,
      "Portable kernels only support inference mode!");

  return _native_batch_norm_legit_no_training_out(
      ctx,
      in,
      weight,
      bias,
      running_mean,
      running_var,
      momentum,
      eps,
      out,
      mean_out,
      invstd_out);
}

std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_no_stats_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    bool training,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& invstd_out) {
  (void)ctx;
  (void)training;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, invstd_out);

  ET_KERNEL_CHECK(
      ctx,
      check_batch_norm_args(
          in,
          weight,
          bias,
          std::optional<Tensor>(),
          std::optional<Tensor>(),
          momentum,
          eps,
          out,
          mean_out,
          invstd_out),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      is_contiguous_dim_order(in.dim_order().data(), in.dim_order().size()),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(in, out, mean_out, invstd_out),
      InvalidArgument,
      ret_val);

  if (weight.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(in, weight.value()),
        InvalidArgument,
        ret_val);
  }

  if (bias.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(in, bias.value()),
        InvalidArgument,
        ret_val);
  }

  ET_KERNEL_CHECK(ctx, in.dim() >= 2, InvalidArgument, ret_val);

  size_t N = in.size(0);
  size_t C = in.size(1);
  size_t inner = getTrailingDims(in, 1);
  size_t elements_per_channel = N * inner;

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(mean_out, {static_cast<SizesType>(C)}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(invstd_out, {static_cast<SizesType>(C)}) == Error::Ok,
      InvalidArgument,
      ret_val);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  ET_DEFINE_OPERATOR_NAME(name, "_native_batch_norm_legit.no_stats_out");

  ET_SWITCH_FLOATHBF16_TYPES(in.scalar_type(), ctx, name, CTYPE, [&] {
    const CTYPE* in_data = in.const_data_ptr<CTYPE>();
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
    CTYPE* mean_data = mean_out.mutable_data_ptr<CTYPE>();
    CTYPE* invstd_data = invstd_out.mutable_data_ptr<CTYPE>();

    // Initialize accumulators to zero before accumulating
    std::memset(mean_data, 0, C * sizeof(CTYPE));
    std::memset(invstd_data, 0, C * sizeof(CTYPE));

    // Compute sum and sum of squares for each channel
    for (size_t b = 0; b < N; ++b) {
      const CTYPE* b_in_data = in_data + b * C * inner;
      for (size_t c = 0; c < C; ++c) {
        const CTYPE* x = b_in_data + c * inner;

        CTYPE sum = reduce_add(x, inner);
        CTYPE sq_sum = vec_powerf(x, inner);

        mean_data[c] += sum;
        invstd_data[c] += sq_sum;
      }
    }

    // Compute mean and invstd for each channel
    const CTYPE elements_per_channel_ct =
        static_cast<CTYPE>(elements_per_channel);
    for (size_t c = 0; c < C; ++c) {
      CTYPE mean = mean_data[c] / elements_per_channel_ct;
      // Var[x] = E[x^2] - E[x]^2
      CTYPE var = invstd_data[c] / elements_per_channel_ct - mean * mean;
      CTYPE invstd = 1.0 / std::sqrt(var + eps);
      mean_data[c] = mean;
      invstd_data[c] = invstd;
    }

    const auto weight_param = ParamReader<CTYPE, name>(ctx, weight);
    const auto bias_param = ParamReader<CTYPE, name>(ctx, bias);
    if (!weight_param.ok() || !bias_param.ok()) {
      return;
    }

    for (size_t i = 0; i < N; ++i) {
      for (size_t c = 0; c < C; ++c) {
        CTYPE mean = mean_data[c];
        CTYPE invstd = invstd_data[c];
        const CTYPE weight_val = weight_param.at_or(c, CTYPE(1));
        const CTYPE bias_val = bias_param.at_or(c, CTYPE(0));
        for (size_t j = 0; j < inner; ++j) {
          *out_data = (*in_data - mean) * invstd * weight_val + bias_val;
          out_data++;
          in_data++;
        }
      }
    }
  });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
