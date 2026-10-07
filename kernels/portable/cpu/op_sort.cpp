/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <c10/util/irange.h>
#include <algorithm>
#include <tuple>
#include <utility>

namespace torch {
namespace executor {
namespace native {
namespace {

bool check_sort_args(
    const Tensor& in,
    int64_t dim,
    Tensor& values,
    Tensor& indices) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, values));
  ET_LOG_AND_RETURN_IF_FALSE(indices.scalar_type() == ScalarType::Long);
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));
  return true;
}

// torch orders NaN above every number, so a NaN lands last ascending and first
// descending.
template <typename T>
bool value_less_than(T x, T y) {
  if constexpr (std::is_integral_v<T>) {
    return x < y;
  }
  return (!utils::isnan_override(x) && utils::isnan_override(y)) || x < y;
}

template <typename CTYPE, typename elem_t = std::pair<CTYPE, int64_t>>
void perform_sort(
    const Tensor& in,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices,
    elem_t* queue) {
  const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
  CTYPE* values_data = values.mutable_data_ptr<CTYPE>();
  int64_t* indices_data = indices.mutable_data_ptr<int64_t>();

  if (in.dim() == 0) {
    values_data[0] = in_data[0];
    indices_data[0] = 0;
    return;
  }

  const size_t outer_size = getLeadingDims(in, dim);
  const size_t dim_size = in.size(dim);
  const size_t dim_stride = in.strides()[dim];
  const size_t outer_stride = dim_size * dim_stride;

  // Explicit typing for the lambdas to satisfy the msvc compiler.
  bool (*elem_greater)(const elem_t&, const elem_t&) =
      [](const elem_t& x, const elem_t& y) -> bool {
    return value_less_than(y.first, x.first);
  };
  bool (*elem_less)(const elem_t&, const elem_t&) =
      [](const elem_t& x, const elem_t& y) -> bool {
    return value_less_than(x.first, y.first);
  };
  bool (*cmp)(const elem_t&, const elem_t&) =
      descending ? elem_greater : elem_less;

  for (const auto outer_idx : c10::irange(outer_size)) {
    const size_t base = outer_idx * outer_stride;
    for (const auto inner_idx : c10::irange(dim_stride)) {
      const size_t base_ix = base + inner_idx;

      for (const auto i : c10::irange(dim_size)) {
        queue[i].first = in_data[base_ix + i * dim_stride];
        queue[i].second = static_cast<int64_t>(i);
      }

      // A stable sort keeps the order of equal values, which satisfies
      // sort.values_stable as well and keeps the result deterministic.
      std::stable_sort(queue, queue + dim_size, cmp);

      for (const auto i : c10::irange(dim_size)) {
        const size_t out_ix = base_ix + i * dim_stride;
        values_data[out_ix] = queue[i].first;
        indices_data[out_ix] = queue[i].second;
      }
    }
  }
}

void* allocate_temp_memory(KernelRuntimeContext& ctx, size_t size) {
  Result<void*> temp_mem_res = ctx.allocate_temp(size);
  return temp_mem_res.ok() ? temp_mem_res.get() : nullptr;
}

} // namespace

std::tuple<Tensor&, Tensor&> sort_values_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  auto out = std::tuple<Tensor&, Tensor&>({values, indices});

  ET_KERNEL_CHECK(
      ctx, check_sort_args(in, dim, values, indices), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(in, values, indices),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  if (dim < 0) {
    dim += nonzero_dim(in);
  }

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(values, in.sizes()) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(indices, in.sizes()) == Error::Ok,
      InvalidArgument,
      out);

  if (in.numel() == 0) {
    return out;
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  ET_DEFINE_OPERATOR_NAME(name, "sort.values");

  bool temp_mem_allocated = false;

  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, name, CTYPE, [&]() {
    using elem_t = std::pair<CTYPE, int64_t>;
    const size_t temp_mem_size = nonempty_size(in, dim) * sizeof(elem_t);

    elem_t* queue = (elem_t*)allocate_temp_memory(ctx, temp_mem_size);
    if (queue == nullptr) {
      return;
    }
    temp_mem_allocated = true;

    perform_sort<CTYPE>(in, dim, descending, values, indices, queue);
  });

  ET_KERNEL_CHECK(ctx, temp_mem_allocated, MemoryAllocationFailed, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
