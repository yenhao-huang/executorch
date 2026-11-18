/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <string_view>
#include <type_traits>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

namespace {

enum class ReduceMode { Sum, Prod, Mean, Amax, Amin };

bool parse_reduce_mode(std::string_view reduce, ReduceMode* mode) {
  if (reduce == "sum") {
    *mode = ReduceMode::Sum;
  } else if (reduce == "prod") {
    *mode = ReduceMode::Prod;
  } else if (reduce == "mean") {
    *mode = ReduceMode::Mean;
  } else if (reduce == "amax") {
    *mode = ReduceMode::Amax;
  } else if (reduce == "amin") {
    *mode = ReduceMode::Amin;
  } else {
    ET_LOG(
        Error,
        "reduce argument must be either sum, prod, mean, amax or amin, got %.*s",
        static_cast<int>(reduce.size()),
        reduce.data());
    return false;
  }
  return true;
}

template <typename CTYPE>
CTYPE reduce_identity(ReduceMode mode) {
  using limits = std::numeric_limits<CTYPE>;
  switch (mode) {
    case ReduceMode::Prod:
      return static_cast<CTYPE>(1);
    case ReduceMode::Amax:
      return limits::has_infinity ? -limits::infinity() : limits::lowest();
    case ReduceMode::Amin:
      return limits::has_infinity ? limits::infinity() : limits::max();
    case ReduceMode::Sum:
    case ReduceMode::Mean:
      break;
  }
  return static_cast<CTYPE>(0);
}

template <typename CTYPE>
CTYPE reduce_pair(ReduceMode mode, CTYPE acc, CTYPE value) {
  switch (mode) {
    case ReduceMode::Sum:
    case ReduceMode::Mean:
      return static_cast<CTYPE>(acc + value);
    case ReduceMode::Prod:
      return static_cast<CTYPE>(acc * value);
    case ReduceMode::Amax:
      return utils::isnan_override(value) || acc < value ? value : acc;
    case ReduceMode::Amin:
      return utils::isnan_override(value) || value < acc ? value : acc;
  }
  return acc;
}

template <typename CTYPE>
CTYPE mean_divide(CTYPE sum, int64_t count) {
  if constexpr (std::is_integral_v<CTYPE>) {
    // ATen keeps the count in the input dtype, so it wraps for narrow types,
    // and replaces a zero count with one.
    CTYPE divisor = static_cast<CTYPE>(count);
    if (divisor == 0) {
      divisor = 1;
    }
    return utils::floor_divide<CTYPE>(sum, divisor);
  } else {
    return static_cast<CTYPE>(sum / static_cast<CTYPE>(count));
  }
}

size_t output_index(
    const Tensor& out,
    int64_t dim,
    int64_t index_value,
    const size_t* ix_coord) {
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  size_t out_coord[kTensorDimensionLimit];
  for (const auto d : c10::irange(out.dim())) {
    out_coord[d] = static_cast<int64_t>(d) == dim
        ? static_cast<size_t>(index_value)
        : ix_coord[d];
  }
  return coordinateToIndex(out, out_coord);
}

template <typename CTYPE, typename ACC>
void scatter_reduce_helper(
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    ReduceMode mode,
    bool include_self,
    const Tensor& out,
    ACC* acc,
    int64_t* counts) {
  const int64_t* const index_data = index.const_data_ptr<int64_t>();
  const CTYPE* const src_data = src.const_data_ptr<CTYPE>();
  // A 0-dim index leaves ix_coord unwritten while src may still be 1-dim.
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  size_t ix_coord[kTensorDimensionLimit] = {};

  if (!include_self) {
    const ACC identity = reduce_identity<ACC>(mode);
    for (const auto ix : c10::irange(index.numel())) {
      indexToCoordinate(index, ix, ix_coord);
      acc[output_index(out, dim, index_data[ix], ix_coord)] = identity;
    }
  }

  for (const auto ix : c10::irange(index.numel())) {
    indexToCoordinate(index, ix, ix_coord);
    const size_t out_ix = output_index(out, dim, index_data[ix], ix_coord);
    const ACC value =
        static_cast<ACC>(src_data[coordinateToIndex(src, ix_coord)]);
    acc[out_ix] = reduce_pair(mode, acc[out_ix], value);
    if (counts != nullptr) {
      counts[out_ix] += 1;
    }
  }

  if (counts != nullptr) {
    for (const auto i : c10::irange(out.numel())) {
      if (counts[i] > 1) {
        acc[i] = mean_divide(acc[i], counts[i]);
      }
    }
  }
}

} // namespace

Tensor& scatter_reduce_two_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    std::string_view reduce,
    bool include_self,
    Tensor& out) {
  ReduceMode mode = ReduceMode::Sum;
  ET_KERNEL_CHECK(ctx, parse_reduce_mode(reduce, &mode), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      check_scatter_add_args(self, dim, index, src, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      !(mode == ReduceMode::Mean && self.scalar_type() == ScalarType::Bool),
      InvalidArgument,
      out,
      "scatter_reduce mean does not support Bool");

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(self, src, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensor_is_default_dim_order(index), InvalidArgument, out);

  if (dim < 0) {
    dim += nonzero_dim(self);
  }

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, self.sizes()) == Error::Ok, InvalidArgument, out);

  // Mean needs the number of contributions per output position, which the
  // output buffer cannot hold alongside the running sum.
  int64_t* counts = nullptr;
  if (mode == ReduceMode::Mean && index.numel() > 0) {
    Result<void*> counts_mem = ctx.allocate_temp(out.numel() * sizeof(int64_t));
    ET_KERNEL_CHECK(ctx, counts_mem.ok(), MemoryAllocationFailed, out);
    counts = static_cast<int64_t*>(counts_mem.get());
    std::fill(counts, counts + out.numel(), include_self ? 1 : 0);
  }

  static constexpr const char op_name[] = "scatter_reduce.two_out";

  ET_SWITCH_REALHBBF16_TYPES(self.scalar_type(), ctx, op_name, CTYPE, [&]() {
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();
    std::memcpy(out_data, self.const_data_ptr<CTYPE>(), self.nbytes());
    if (index.numel() == 0) {
      return;
    }
    // Accumulate Half/BFloat16 in float, as ATen does.
    if constexpr (c10::is_reduced_floating_point_v<CTYPE>) {
      Result<void*> acc_mem = ctx.allocate_temp(out.numel() * sizeof(float));
      ET_KERNEL_CHECK(ctx, acc_mem.ok(), MemoryAllocationFailed, );
      float* const acc = static_cast<float*>(acc_mem.get());
      std::copy(out_data, out_data + out.numel(), acc);
      scatter_reduce_helper<CTYPE>(
          dim, index, src, mode, include_self, out, acc, counts);
      std::copy(acc, acc + out.numel(), out_data);
    } else {
      scatter_reduce_helper<CTYPE>(
          dim, index, src, mode, include_self, out, out_data, counts);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
