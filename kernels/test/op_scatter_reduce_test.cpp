/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TempMemoryAllocator.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

#include <string_view>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TempMemoryAllocator;
using torch::executor::testing::TensorFactory;

Tensor& op_scatter_reduce_two_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    std::string_view reduce,
    bool include_self,
    Tensor& out) {
  TempMemoryAllocator allocator = TempMemoryAllocator();
  executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext context(
      nullptr, &allocator);
  return torch::executor::aten::scatter_reduce_outf(
      context, self, dim, index, src, reduce, include_self, out);
}

class OpScatterReduceTwoOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }

  // All references come from torch.scatter_reduce.
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
  EXPECT_TENSOR_CLOSE(
      out, tf.make({2, 3}, {5.5, 52.0 / 3, 3, 38, 22.5, 6}));
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

  // amax and amin start from the integer bounds rather than infinities.
  op_scatter_reduce_two_out(self, 0, index, src, "amax", false, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {2, 20, 5, 40}));

  op_scatter_reduce_two_out(self, 0, index, src, "amin", false, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {1, 20, 5, 40}));
}

TEST_F(OpScatterReduceTwoOutTest, EmptyIndexCopiesSelf) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf.make({4}, {10, 20, 30, 40});
  Tensor index = tf_long.make({0}, {});
  Tensor src = tf.make({0}, {});
  Tensor out = tf.zeros({4});

  op_scatter_reduce_two_out(self, 0, index, src, "sum", true, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {10, 20, 30, 40}));
}
