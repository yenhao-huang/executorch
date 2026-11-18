/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/memory_allocator.h>

#include <gtest/gtest.h>

#include <cmath>
#include <string_view>
#include <vector>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::MemoryAllocator;
using torch::executor::testing::TensorFactory;

// Expected values come from torch.scatter_reduce.
class OpScatterReduceTwoOutTest : public OperatorTest {
 protected:
  OpScatterReduceTwoOutTest() : allocator_(sizeof(buffer_), buffer_) {
    context_ = torch::executor::KernelRuntimeContext(nullptr, &allocator_);
  }

  Tensor& op_scatter_reduce_two_out(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& src,
      std::string_view reduce,
      bool include_self,
      Tensor& out) {
    return torch::executor::aten::scatter_reduce_outf(
        context_, self, dim, index, src, reduce, include_self, out);
  }

  void check_1d(
      std::string_view reduce,
      bool include_self,
      const std::vector<float>& expected) {
    TensorFactory<ScalarType::Float> tf;
    TensorFactory<ScalarType::Long> tf_long;

    Tensor self = tf.make({4}, {10, 20, 30, 40});
    Tensor index = tf_long.make({3}, {0, 0, 2});
    Tensor src = tf.make({3}, {1, 2, 5});
    Tensor out = tf.zeros({4});

    op_scatter_reduce_two_out(self, 0, index, src, reduce, include_self, out);
    EXPECT_TENSOR_CLOSE(out, tf.make({4}, expected));
  }

  void check_index_smaller_than_src(
      std::string_view reduce,
      bool include_self,
      const std::vector<float>& expected) {
    TensorFactory<ScalarType::Float> tf;
    TensorFactory<ScalarType::Long> tf_long;

    Tensor self = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor index = tf_long.make({2, 2}, {2, 0, 1, 1});
    Tensor src = tf.make({2, 3}, {10, 20, 30, 40, 50, 60});
    Tensor out = tf.zeros({2, 3});

    op_scatter_reduce_two_out(self, -1, index, src, reduce, include_self, out);
    EXPECT_TENSOR_CLOSE(out, tf.make({2, 3}, expected));
  }

  alignas(16) uint8_t buffer_[1024];
  MemoryAllocator allocator_;
};

TEST_F(OpScatterReduceTwoOutTest, SumIncludesSelf) {
  check_1d("sum", true, {13, 20, 35, 40});
}

TEST_F(OpScatterReduceTwoOutTest, SumWithoutSelf) {
  check_1d("sum", false, {3, 20, 5, 40});
}

TEST_F(OpScatterReduceTwoOutTest, ProdIncludesSelf) {
  check_1d("prod", true, {20, 20, 150, 40});
}

TEST_F(OpScatterReduceTwoOutTest, ProdWithoutSelf) {
  check_1d("prod", false, {2, 20, 5, 40});
}

TEST_F(OpScatterReduceTwoOutTest, MeanIncludesSelf) {
  check_1d("mean", true, {13.0 / 3, 20, 17.5, 40});
}

TEST_F(OpScatterReduceTwoOutTest, MeanWithoutSelf) {
  check_1d("mean", false, {1.5, 20, 5, 40});
}

TEST_F(OpScatterReduceTwoOutTest, AmaxIncludesSelf) {
  check_1d("amax", true, {10, 20, 30, 40});
}

TEST_F(OpScatterReduceTwoOutTest, AmaxWithoutSelf) {
  check_1d("amax", false, {2, 20, 5, 40});
}

TEST_F(OpScatterReduceTwoOutTest, AminIncludesSelf) {
  check_1d("amin", true, {1, 20, 5, 40});
}

TEST_F(OpScatterReduceTwoOutTest, AminWithoutSelf) {
  check_1d("amin", false, {1, 20, 5, 40});
}

TEST_F(OpScatterReduceTwoOutTest, TwoDimensional) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor index = tf_long.make({2, 3}, {0, 1, 1, 1, 0, 0});
  Tensor src = tf.make({2, 3}, {10, 20, 30, 40, 50, 60});
  Tensor out = tf.zeros({2, 3});

  op_scatter_reduce_two_out(self, 0, index, src, "sum", true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 3}, {11, 52, 63, 44, 25, 36}));

  op_scatter_reduce_two_out(self, 1, index, src, "sum", true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 3}, {11, 52, 3, 114, 45, 6}));

  op_scatter_reduce_two_out(self, 0, index, src, "amax", true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 3}, {10, 50, 60, 40, 20, 30}));

  op_scatter_reduce_two_out(self, 1, index, src, "mean", true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 3}, {5.5, 52.0 / 3, 3, 38, 22.5, 6}));
}

TEST_F(OpScatterReduceTwoOutTest, IndexSmallerThanSrcNegativeDim) {
  check_index_smaller_than_src("sum", true, {21, 2, 13, 4, 95, 6});
  check_index_smaller_than_src("sum", false, {20, 2, 10, 4, 90, 6});
  check_index_smaller_than_src("prod", true, {20, 2, 30, 4, 10000, 6});
  check_index_smaller_than_src("prod", false, {20, 2, 10, 4, 2000, 6});
  check_index_smaller_than_src("mean", true, {10.5, 2, 6.5, 4, 95.0 / 3, 6});
  check_index_smaller_than_src("mean", false, {20, 2, 10, 4, 45, 6});
  check_index_smaller_than_src("amax", true, {20, 2, 10, 4, 50, 6});
  check_index_smaller_than_src("amax", false, {20, 2, 10, 4, 50, 6});
  check_index_smaller_than_src("amin", true, {1, 2, 3, 4, 5, 6});
  check_index_smaller_than_src("amin", false, {20, 2, 10, 4, 40, 6});
}

TEST_F(OpScatterReduceTwoOutTest, IndexSmallerThanSrcAlongDim0) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor index = tf_long.make({1, 2}, {1, 0});
  Tensor src = tf.make({2, 3}, {10, 20, 30, 40, 50, 60});
  Tensor out = tf.zeros({2, 3});

  op_scatter_reduce_two_out(self, 0, index, src, "prod", false, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 3}, {1, 20, 3, 10, 5, 6}));
}

TEST_F(OpScatterReduceTwoOutTest, IntegerDtypeWithoutSelf) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf.make({4}, {10, 20, 30, 40});
  Tensor index = tf_long.make({3}, {0, 0, 2});
  Tensor src = tf.make({3}, {1, 2, 5});
  Tensor out = tf.zeros({4});

  op_scatter_reduce_two_out(self, 0, index, src, "sum", false, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {3, 20, 5, 40}));

  op_scatter_reduce_two_out(self, 0, index, src, "prod", false, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {2, 20, 5, 40}));

  op_scatter_reduce_two_out(self, 0, index, src, "amax", false, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {2, 20, 5, 40}));

  op_scatter_reduce_two_out(self, 0, index, src, "amin", false, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {1, 20, 5, 40}));
}

TEST_F(OpScatterReduceTwoOutTest, IntegerMeanRoundsTowardNegativeInfinity) {
  TensorFactory<ScalarType::Long> tf;

  Tensor self = tf.make({4}, {-7, 7, 5, -4});
  Tensor index = tf.make({5}, {0, 0, 1, 1, 3});
  Tensor src = tf.make({5}, {-2, 1, 2, 3, 1});
  Tensor out = tf.zeros({4});

  op_scatter_reduce_two_out(self, 0, index, src, "mean", true, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {-3, 4, 5, -2}));

  op_scatter_reduce_two_out(self, 0, index, src, "mean", false, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {-1, 2, 5, 1}));
}

TEST_F(OpScatterReduceTwoOutTest, AmaxAminPropagateNan) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  const float nan = NAN;
  Tensor self = tf.make({3}, {1, nan, 3});
  Tensor index = tf_long.make({4}, {0, 0, 1, 2});
  Tensor src = tf.make({4}, {nan, 4, 5, 2});
  Tensor out = tf.zeros({3});

  op_scatter_reduce_two_out(self, 0, index, src, "amax", true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({3}, {nan, nan, 3}));

  op_scatter_reduce_two_out(self, 0, index, src, "amin", true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({3}, {nan, nan, 2}));
}

TEST_F(OpScatterReduceTwoOutTest, HalfAndBFloat16) {
  TensorFactory<ScalarType::Half> tf_half;
  TensorFactory<ScalarType::BFloat16> tf_bf16;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor index = tf_long.make({3}, {0, 0, 2});

  Tensor half_out = tf_half.zeros({3});
  op_scatter_reduce_two_out(
      tf_half.make({3}, {1, 2, 3}),
      0,
      index,
      tf_half.make({3}, {1.5, 2.5, 4}),
      "mean",
      true,
      half_out);
  EXPECT_TENSOR_CLOSE(half_out, tf_half.make({3}, {1.666992, 2, 3.5}));

  Tensor bf16_out = tf_bf16.zeros({3});
  op_scatter_reduce_two_out(
      tf_bf16.make({3}, {1, 2, 3}),
      0,
      index,
      tf_bf16.make({3}, {-1.5, -2.5, 4}),
      "amax",
      false,
      bf16_out);
  EXPECT_TENSOR_CLOSE(bf16_out, tf_bf16.make({3}, {-1.5, 2, 4}));
}

TEST_F(OpScatterReduceTwoOutTest, HalfAndBFloat16AccumulateInFloat) {
  TensorFactory<ScalarType::Half> tf_half;
  TensorFactory<ScalarType::BFloat16> tf_bf16;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor index = tf_long.zeros({4096});
  Tensor half_out = tf_half.zeros({3});

  op_scatter_reduce_two_out(
      tf_half.zeros({3}),
      0,
      index,
      tf_half.ones({4096}),
      "sum",
      true,
      half_out);
  EXPECT_TENSOR_CLOSE(half_out, tf_half.make({3}, {4096, 0, 0}));

  op_scatter_reduce_two_out(
      tf_half.zeros({3}),
      0,
      index,
      tf_half.ones({4096}),
      "mean",
      true,
      half_out);
  EXPECT_TENSOR_CLOSE(half_out, tf_half.make({3}, {1, 0, 0}));

  op_scatter_reduce_two_out(
      tf_half.ones({3}),
      0,
      index,
      tf_half.full({4096}, 1.001),
      "prod",
      false,
      half_out);
  EXPECT_TENSOR_CLOSE(half_out, tf_half.make({3}, {54.5, 1, 1}));

  Tensor bf16_out = tf_bf16.zeros({3});
  op_scatter_reduce_two_out(
      tf_bf16.zeros({3}),
      0,
      tf_long.zeros({1000}),
      tf_bf16.ones({1000}),
      "sum",
      true,
      bf16_out);
  EXPECT_TENSOR_CLOSE(bf16_out, tf_bf16.make({3}, {1000, 0, 0}));
}

TEST_F(OpScatterReduceTwoOutTest, NarrowIntegerMeanCountWraps) {
  TensorFactory<ScalarType::Byte> tf;
  TensorFactory<ScalarType::Char> tf_char;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf.make({2}, {7, 3});
  Tensor out = tf.zeros({2});

  op_scatter_reduce_two_out(
      self, 0, tf_long.zeros({256}), tf.full({256}, 2), "mean", false, out);
  EXPECT_TENSOR_EQ(out, tf.make({2}, {0, 3}));

  op_scatter_reduce_two_out(
      self, 0, tf_long.zeros({255}), tf.full({255}, 2), "mean", true, out);
  EXPECT_TENSOR_EQ(out, tf.make({2}, {5, 3}));

  Tensor char_out = tf_char.zeros({1});
  op_scatter_reduce_two_out(
      tf_char.zeros({1}),
      0,
      tf_long.zeros({200}),
      tf_char.ones({200}),
      "mean",
      false,
      char_out);
  EXPECT_TENSOR_EQ(char_out, tf_char.make({1}, {1}));
}

TEST_F(OpScatterReduceTwoOutTest, ZeroDimIndexWithOneDimSrc) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor out = tf.zeros({3});

  op_scatter_reduce_two_out(
      tf.zeros({3}),
      0,
      tf_long.make({}, {1}),
      tf.make({1}, {5}),
      "sum",
      true,
      out);
  EXPECT_TENSOR_CLOSE(out, tf.make({3}, {0, 5, 0}));

  op_scatter_reduce_two_out(
      tf.ones({3}),
      0,
      tf_long.make({}, {2}),
      tf.make({1}, {5}),
      "prod",
      false,
      out);
  EXPECT_TENSOR_CLOSE(out, tf.make({3}, {1, 1, 5}));
}

TEST_F(OpScatterReduceTwoOutTest, BoolWithoutSelf) {
  TensorFactory<ScalarType::Bool> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf.make({3}, {true, false, true});
  Tensor index = tf_long.make({2}, {1, 2});
  Tensor src = tf.make({2}, {true, false});
  Tensor out = tf.zeros({3});

  for (const std::string_view reduce : {"sum", "prod", "amax", "amin"}) {
    op_scatter_reduce_two_out(self, 0, index, src, reduce, false, out);
    EXPECT_TENSOR_EQ(out, tf.make({3}, {true, true, false}));
  }
}

TEST_F(OpScatterReduceTwoOutTest, ZeroDimTensors) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf.make({}, {3});
  Tensor index = tf_long.make({}, {0});
  Tensor src = tf.make({}, {5});
  Tensor out = tf.zeros({});

  op_scatter_reduce_two_out(self, 0, index, src, "prod", true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({}, {15}));

  op_scatter_reduce_two_out(self, -1, index, src, "mean", true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({}, {4}));

  op_scatter_reduce_two_out(self, 0, index, src, "amin", false, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({}, {5}));
}

TEST_F(OpScatterReduceTwoOutTest, EmptyIndexCopiesSelf) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf.make({4}, {10, 20, 30, 40});
  Tensor index = tf_long.make({0}, {});
  Tensor src = tf.make({0}, {});
  Tensor out = tf.zeros({4});

  op_scatter_reduce_two_out(self, 0, index, src, "sum", true, out);
  EXPECT_TENSOR_EQ(out, self);

  op_scatter_reduce_two_out(self, 0, index, src, "mean", false, out);
  EXPECT_TENSOR_EQ(out, self);
}

TEST_F(OpScatterReduceTwoOutTest, InvalidReduceDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf.zeros({3});
  Tensor index = tf_long.make({1}, {0});
  Tensor src = tf.ones({1});
  Tensor out = tf.zeros({3});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_scatter_reduce_two_out(self, 0, index, src, "foo", true, out));
}

TEST_F(OpScatterReduceTwoOutTest, OutOfBoundsIndexDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf.zeros({3});
  Tensor index = tf_long.make({1}, {3});
  Tensor src = tf.ones({1});
  Tensor out = tf.zeros({3});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_scatter_reduce_two_out(self, 0, index, src, "sum", true, out));
}

TEST_F(OpScatterReduceTwoOutTest, BoolMeanDies) {
  TensorFactory<ScalarType::Bool> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf.zeros({3});
  Tensor index = tf_long.make({1}, {0});
  Tensor src = tf.ones({1});
  Tensor out = tf.zeros({3});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_scatter_reduce_two_out(self, 0, index, src, "mean", true, out));
}

TEST_F(OpScatterReduceTwoOutTest, MeanWithoutTempAllocatorDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernels do not use the temp allocator";
  }
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf.zeros({3});
  Tensor index = tf_long.make({1}, {0});
  Tensor src = tf.ones({1});
  Tensor out = tf.zeros({3});

  context_ = torch::executor::KernelRuntimeContext();
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_scatter_reduce_two_out(self, 0, index, src, "mean", true, out));
}
