/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>
#include <initializer_list>
#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::TensorShapeDynamism;
using executorch::runtime::testing::TensorFactory;

class OpAdaptiveAvgPool1DOutTest : public OperatorTest {
 protected:
  Tensor& pool(
      const Tensor& in,
      std::initializer_list<int64_t> output_size,
      Tensor& out) {
    return torch::executor::aten::adaptive_avg_pool1d_outf(
        context_, in, {output_size.begin(), output_size.size()}, out);
  }

  template <ScalarType DTYPE>
  void test_non_divisible_dtype() {
    TensorFactory<DTYPE> tf;
    auto in = tf.make({1, 1, 7}, {-3, -2, -1, 0, 1, 2, 3});
    auto out = tf.zeros({1, 1, 3});
    pool(in, {3}, out);
    EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 3}, {-2, 0, 2}));
  }
};

TEST_F(OpAdaptiveAvgPool1DOutTest, DivisibleBatched) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make(
      {1, 2, 6},
      {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
  auto out = tf.zeros({1, 2, 3});
  pool(in, {3}, out);
  EXPECT_TENSOR_CLOSE(
      out, tf.make({1, 2, 3}, {-2.5, -0.5, 1.5, 4.5, 6.5, 8.5}));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, NonDivisibleUnbatched) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({2, 7}, {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto out = tf.zeros({2, 3});
  pool(in, {3}, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 3}, {-2.0, 0.0, 2.0, 5.0, 7.0, 9.0}));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, NonDivisibleMultiBatch) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({2, 1, 10}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                 -1, -2, -3, -4, -5, -6, -7, -8, -9, -10});
  auto out = tf.zeros({2, 1, 4});
  pool(in, {4}, out);
  EXPECT_TENSOR_CLOSE(
      out, tf.make({2, 1, 4}, {2.0, 4.0, 7.0, 9.0, -2.0, -4.0, -7.0, -9.0}));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, OutputLargerThanInput) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 3}, {1.0, 4.0, 9.0});
  auto out = tf.zeros({1, 1, 5});
  pool(in, {5}, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 5}, {1.0, 2.5, 4.0, 6.5, 9.0}));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, GlobalPool) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 4}, {1.0, 4.0, 9.0, 16.0});
  auto out = tf.zeros({1, 1, 1});
  pool(in, {1}, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 1}, {7.5}));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, SameSize) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 4}, {1.0, 4.0, 9.0, 16.0});
  auto out = tf.zeros({1, 1, 4});
  pool(in, {4}, out);
  EXPECT_TENSOR_EQ(out, in);
}

TEST_F(OpAdaptiveAvgPool1DOutTest, EmptyBatch) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({0, 2, 4}, {});
  auto out = tf.zeros({0, 2, 2});
  pool(in, {2}, out);
  EXPECT_TENSOR_EQ(out, tf.make({0, 2, 2}, {}));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, EmptyChannels) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 0, 4}, {});
  auto out = tf.zeros({1, 0, 2});
  pool(in, {2}, out);
  EXPECT_TENSOR_EQ(out, tf.make({1, 0, 2}, {}));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, ZeroOutputSize) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 2, 4});
  auto out = tf.zeros({1, 2, 0});
  pool(in, {0}, out);
  EXPECT_TENSOR_EQ(out, tf.make({1, 2, 0}, {}));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, DoubleDtype) {
  TensorFactory<ScalarType::Double> tf;
  auto in = tf.make({1, 1, 3}, {0.1, 0.2, 0.3});
  auto out = tf.zeros({1, 1, 2});
  pool(in, {2}, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 2}, {0.15, 0.25}));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, FloatTypes) {
  test_non_divisible_dtype<ScalarType::Float>();
  test_non_divisible_dtype<ScalarType::Double>();
  test_non_divisible_dtype<ScalarType::Half>();
  test_non_divisible_dtype<ScalarType::BFloat16>();
}

TEST_F(OpAdaptiveAvgPool1DOutTest, DynamicOutput) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 4}, {1.0, 4.0, 9.0, 16.0});
  auto out = tf.zeros({1, 1, 5}, TensorShapeDynamism::DYNAMIC_BOUND);
  pool(in, {3}, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 3}, {2.5, 6.5, 12.5}));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, WrongInputRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({4});
  auto out = tf.zeros({2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, out));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, FourDimensionalInput) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 1, 4});
  auto out = tf.zeros({1, 1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, out));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, EmptyLength) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 0});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, out));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, NegativeOutputSize) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 4});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {-1}, out));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, OutputSizeExceedsSizesType) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 4});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {4294967298}, out));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, WrongOutputSizeRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 4});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2, 2}, out));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, EmptyOutputSize) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 4});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {}, out));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, MismatchedDtype) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> other;
  auto in = tf.ones({1, 1, 4});
  auto out = other.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, out));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, UnsupportedLongDtype) {
  TensorFactory<ScalarType::Long> tf;
  auto in = tf.ones({1, 1, 4});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, out));
}

#ifndef USE_ATEN_LIB
TEST_F(OpAdaptiveAvgPool1DOutTest, HalfAccumulatesInFloat) {
  TensorFactory<ScalarType::Half> tf;
  auto in = tf.full({1, 1, 3}, 60000);
  auto out = tf.zeros({1, 1, 1});
  pool(in, {1}, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 1}, 60000));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, WrongOutputRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 4});
  auto out = tf.zeros({1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, out));
}

TEST_F(OpAdaptiveAvgPool1DOutTest, NonDefaultDimOrder) {
  TensorFactory<ScalarType::Float> tf;
  auto in =
      tf.make_with_dimorder({2, 2, 2}, std::vector<float>(8, 1), {0, 2, 1});
  auto out = tf.zeros({2, 2, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {1}, out));
}
#endif
