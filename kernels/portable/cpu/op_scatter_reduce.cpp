/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <string_view>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using string_view = std::string_view;

namespace {

enum class ReduceMode { Sum, Prod, Mean, Amax, Amin };

bool parse_reduce_mode(string_view reduce, ReduceMode* mode) {
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
        "Unsupported reduce mode '%.*s'; expected one of sum, prod, mean, amax, amin",
        static_cast<int>(reduce.size()),
        reduce.data());
    return false;
  }
  return true;
}

// The value a position starts from when `self` is not included in the
// reduction.
template <typename CTYPE>
CTYPE identity_for(ReduceMode mode) {
  switch (mode) {
    case ReduceMode::Prod:
      return static_cast<CTYPE>(1);
    case ReduceMode::Amax:
      return std::numeric_limits<CTYPE>::has_infinity
          ? -std::numeric_limits<CTYPE>::infinity()
          : std::numeric_limits<CTYPE>::lowest();
    case ReduceMode::Amin:
      return std::numeric_limits<CTYPE>::has_infinity
          ? std::numeric_limits<CTYPE>::infinity()
          : std::numeric_limits<CTYPE>::max();
    case ReduceMode::Sum:
    case ReduceMode::Mean:
    default:
      return static_cast<CTYPE>(0);
  }
}

// The flat indices in `src` and `out` that element `ix` of `index` relates.
void resolve_indices(
    const Tensor& index,
    const int64_t* index_data,
    const Tensor& src,
    const Tensor& out,
    int64_t dim,
    size_t ix,
    size_t* src_ix,
    size_t* out_ix) {
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  size_t ix_coord[kTensorDimensionLimit];
  indexToCoordinate(index, ix, ix_coord);

  *src_ix = coordinateToIndex(src, ix_coord);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  size_t out_coord[kTensorDimensionLimit];
  for (const auto d : c10::irange(out.dim())) {
    out_coord[d] = static_cast<int64_t>(d) == dim
        ? static_cast<size_t>(index_data[ix])
        : ix_coord[d];
  }
  *out_ix = coordinateToIndex(out, out_coord);
}

template <typename CTYPE>
void scatter_reduce_helper(
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    ReduceMode mode,
    bool include_self,
    Tensor& out,
    int64_t* counts) {
  const int64_t* const index_data = index.const_data_ptr<int64_t>();
  const CTYPE* const src_data = src.const_data_ptr<CTYPE>();
  CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

  // Count how many src values reach each output position. A position that gets
  // none keeps its `self` value, whatever include_self says.
  std::fill(counts, counts + out.numel(), int64_t{0});
  for (const auto ix : c10::irange(index.numel())) {
    size_t src_ix = 0;
    size_t out_ix = 0;
    resolve_indices(index, index_data, src, out, dim, ix, &src_ix, &out_ix);
    counts[out_ix] += 1;
  }

  if (!include_self) {
    const CTYPE identity = identity_for<CTYPE>(mode);
    for (const auto i : c10::irange(out.numel())) {
      if (counts[i] > 0) {
        out_data[i] = identity;
      }
    }
  }

  for (const auto ix : c10::irange(index.numel())) {
    size_t src_ix = 0;
    size_t out_ix = 0;
    resolve_indices(index, index_data, src, out, dim, ix, &src_ix, &out_ix);
    const CTYPE value = src_data[src_ix];
    CTYPE& target = out_data[out_ix];
    switch (mode) {
      case ReduceMode::Sum:
      case ReduceMode::Mean:
        target += value;
        break;
      case ReduceMode::Prod:
        target *= value;
        break;
      case ReduceMode::Amax:
        target = std::max(target, value);
        break;
      case ReduceMode::Amin:
        target = std::min(target, value);
        break;
    }
  }

  if (mode == ReduceMode::Mean) {
    for (const auto i : c10::irange(out.numel())) {
      const int64_t divisor = counts[i] + (include_self ? 1 : 0);
      if (counts[i] > 0 && divisor > 1) {
        out_data[i] = static_cast<CTYPE>(
            out_data[i] / static_cast<CTYPE>(divisor));
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
    string_view reduce,
    bool include_self,
    Tensor& out) {
  ReduceMode mode = ReduceMode::Sum;
  ET_KERNEL_CHECK(ctx, parse_reduce_mode(reduce, &mode), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      check_scatter_add_args(self, dim, index, src, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(self, src, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensor_is_default_dim_order(index), InvalidArgument, out);

  if (dim < 0) {
    dim += nonzero_dim(self);
  }

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, self.sizes()) == Error::Ok, InvalidArgument, out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  ET_DEFINE_OPERATOR_NAME(name, "scatter_reduce.two_out");

  bool temp_mem_allocated = false;

  ET_SWITCH_REALHBBF16_TYPES(self.scalar_type(), ctx, name, CTYPE, [&]() {
    std::memcpy(
        out.mutable_data_ptr<CTYPE>(), self.const_data_ptr<CTYPE>(), self.nbytes());

    if (index.numel() == 0) {
      temp_mem_allocated = true;
      return;
    }

    // One count per output element decides which positions the reduction
    // touches, which is what include_self and the mean divisor need.
    Result<void*> counts_mem =
        ctx.allocate_temp(out.numel() * sizeof(int64_t));
    if (!counts_mem.ok()) {
      return;
    }
    temp_mem_allocated = true;

    scatter_reduce_helper<CTYPE>(
        dim,
        index,
        src,
        mode,
        include_self,
        out,
        static_cast<int64_t*>(counts_mem.get()));
  });

  ET_KERNEL_CHECK(ctx, temp_mem_allocated, MemoryAllocationFailed, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
