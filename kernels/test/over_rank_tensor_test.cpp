/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Nothing in the runtime caps a tensor's rank, but kernels delinearize indexes
// into stack arrays of kTensorDimensionLimit entries. A tensor of higher rank
// used to overrun those arrays — gather and scatter segfaulted outright. Every
// kernel that indexes such an array has to reject the tensor instead.
// See https://github.com/pytorch/executorch/issues/8237.

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>

#include <gtest/gtest.h>

#include <vector>

using executorch::aten::ArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::kTensorDimensionLimit;
using torch::executor::testing::TensorFactory;

namespace {
constexpr size_t kOverRank = kTensorDimensionLimit + 1;

std::vector<int32_t> over_rank_sizes() {
  std::vector<int32_t> sizes(kOverRank, 1);
  sizes[kOverRank - 1] = 4;
  return sizes;
}
} // namespace

class OverRankTensorTest : public OperatorTest {};

TEST_F(OverRankTensorTest, GatherRejectsOverRankInput) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;
  const auto sizes = over_rank_sizes();

  Tensor in = tf.ones(sizes);
  Tensor index = tf_long.zeros(sizes);
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      torch::executor::aten::gather_outf(context_, in, 0, index, false, out));
}

TEST_F(OverRankTensorTest, ScatterAddRejectsOverRankInput) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;
  const auto sizes = over_rank_sizes();

  Tensor self = tf.ones(sizes);
  Tensor index = tf_long.zeros(sizes);
  Tensor src = tf.ones(sizes);
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      torch::executor::aten::scatter_add_outf(
          context_, self, 0, index, src, out));
}

TEST_F(OverRankTensorTest, ScatterRejectsOverRankInput) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;
  const auto sizes = over_rank_sizes();

  Tensor self = tf.ones(sizes);
  Tensor index = tf_long.zeros(sizes);
  Tensor src = tf.ones(sizes);
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      torch::executor::aten::scatter_outf(context_, self, 0, index, src, out));
}

TEST_F(OverRankTensorTest, IndexSelectRejectsOverRankInput) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;
  const auto sizes = over_rank_sizes();

  Tensor in = tf.ones(sizes);
  Tensor index = tf_long.zeros({1});
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      torch::executor::aten::index_select_outf(context_, in, 0, index, out));
}

TEST_F(OverRankTensorTest, SliceCopyRejectsOverRankInput) {
  TensorFactory<ScalarType::Float> tf;
  const auto sizes = over_rank_sizes();

  Tensor in = tf.ones(sizes);
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      torch::executor::aten::slice_copy_outf(
          context_, in, kOverRank - 1, 0, 4, 1, out));
}

TEST_F(OverRankTensorTest, TransposeCopyRejectsOverRankInput) {
  TensorFactory<ScalarType::Float> tf;
  const auto sizes = over_rank_sizes();

  Tensor in = tf.ones(sizes);
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      torch::executor::aten::transpose_copy_outf(context_, in, 0, 1, out));
}

TEST_F(OverRankTensorTest, CatRejectsOverRankInput) {
  TensorFactory<ScalarType::Float> tf;
  const auto sizes = over_rank_sizes();

  Tensor in = tf.ones(sizes);
  Tensor out = tf.zeros(sizes);
  std::vector<Tensor> to_cat{in};

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      torch::executor::aten::cat_outf(
          context_, ArrayRef<Tensor>(to_cat.data(), to_cat.size()), 0, out));
}

TEST_F(OverRankTensorTest, ConstantPadNdRejectsOverRankInput) {
  TensorFactory<ScalarType::Float> tf;
  const auto sizes = over_rank_sizes();

  Tensor in = tf.ones(sizes);
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      torch::executor::aten::constant_pad_nd_outf(
          context_, in, ArrayRef<int64_t>(), 0, out));
}
