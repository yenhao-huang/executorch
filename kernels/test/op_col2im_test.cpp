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

using executorch::aten::Half;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::TensorShapeDynamism;
using executorch::runtime::testing::TensorFactory;

class OpCol2ImOutTest : public OperatorTest {
 protected:
  Tensor& col2im(
      const Tensor& in,
      std::initializer_list<int64_t> output_size,
      std::initializer_list<int64_t> kernel_size,
      std::initializer_list<int64_t> dilation,
      std::initializer_list<int64_t> padding,
      std::initializer_list<int64_t> stride,
      Tensor& out) {
    return torch::executor::aten::col2im_outf(
        context_,
        in,
        {output_size.begin(), output_size.size()},
        {kernel_size.begin(), kernel_size.size()},
        {dilation.begin(), dilation.size()},
        {padding.begin(), padding.size()},
        {stride.begin(), stride.size()},
        out);
  }

  template <ScalarType DTYPE>
  void test_overlapping_blocks() {
    TensorFactory<DTYPE> tf;
    auto in = tf.make(
        {1, 4, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto out = tf.zeros({1, 1, 3, 3});
    col2im(in, {3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, out);
    EXPECT_TENSOR_EQ(
        out, tf.make({1, 1, 3, 3}, {0, 5, 5, 10, 30, 20, 10, 25, 15}));
  }
};

TEST_F(OpCol2ImOutTest, AllFloatDtypes) {
#define TEST_ENTRY(ctype, dtype) test_overlapping_blocks<ScalarType::dtype>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpCol2ImOutTest, Unbatched) {
  TensorFactory<ScalarType::Float> tf;
  auto in =
      tf.make({4, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto out = tf.zeros({1, 3, 3});
  col2im(in, {3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, out);
  EXPECT_TENSOR_EQ(out, tf.make({1, 3, 3}, {0, 5, 5, 10, 30, 20, 10, 25, 15}));
}

TEST_F(OpCol2ImOutTest, BatchedMultiChannelStride) {
  TensorFactory<ScalarType::Float> tf;
  std::vector<float> in_data(64);
  for (size_t i = 0; i < in_data.size(); ++i) {
    in_data[i] = (static_cast<float>(i) - 30.0f) / 4.0f;
  }
  auto in = tf.make({2, 8, 4}, in_data);
  auto out = tf.zeros({2, 2, 4, 4});
  col2im(in, {4, 4}, {2, 2}, {1, 1}, {0, 0}, {2, 2}, out);
  EXPECT_TENSOR_CLOSE(
      out,
      tf.make(
          {2, 2, 4, 4},
          {-7.5,  -6.5,  -7.25, -6.25, -5.5,  -4.5,  -5.25, -4.25, -7.0,  -6.0,
           -6.75, -5.75, -5.0,  -4.0,  -4.75, -3.75, -3.5,  -2.5,  -3.25, -2.25,
           -1.5,  -0.5,  -1.25, -0.25, -3.0,  -2.0,  -2.75, -1.75, -1.0,  0.0,
           -0.75, 0.25,  0.5,   1.5,   0.75,  1.75,  2.5,   3.5,   2.75,  3.75,
           1.0,   2.0,   1.25,  2.25,  3.0,   4.0,   3.25,  4.25,  4.5,   5.5,
           4.75,  5.75,  6.5,   7.5,   6.75,  7.75,  5.0,   6.0,   5.25,  6.25,
           7.0,   8.0,   7.25,  8.25}));
}

TEST_F(OpCol2ImOutTest, PaddingDilationStride) {
  TensorFactory<ScalarType::Float> tf;
  std::vector<float> in_data(72);
  for (size_t i = 0; i < in_data.size(); ++i) {
    in_data[i] = static_cast<float>(i % 11) - 5.0f;
  }
  auto in = tf.make({1, 6, 12}, in_data);
  auto out = tf.zeros({1, 1, 4, 5});
  col2im(in, {4, 5}, {2, 3}, {2, 1}, {1, 1}, {1, 2}, out);
  EXPECT_TENSOR_EQ(
      out, tf.make({1, 1, 4, 5}, {-1, -1, 0,  1,  1, 1, 4, 3,  8,  5,
                                  7,  5,  -2, -2, 0, 5, 0, -5, -9, -4}));
}

TEST_F(OpCol2ImOutTest, StrideLargerThanKernelLeavesGaps) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 4}, {1, 2, 3, 4});
  auto out = tf.ones({1, 1, 3, 3});
  col2im(in, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {2, 2}, out);
  EXPECT_TENSOR_EQ(out, tf.make({1, 1, 3, 3}, {1, 0, 2, 0, 0, 0, 3, 0, 4}));
}

TEST_F(OpCol2ImOutTest, EmptyBatch) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({0, 4, 4}, {});
  auto out = tf.zeros({0, 1, 3, 3});
  col2im(in, {3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, out);
  EXPECT_TENSOR_EQ(out, tf.make({0, 1, 3, 3}, {}));
}

TEST_F(OpCol2ImOutTest, DynamicOutput) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 4, 4});
  auto out = tf.zeros({2, 2, 4, 4}, TensorShapeDynamism::DYNAMIC_BOUND);
  col2im(in, {3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, out);
  EXPECT_TENSOR_EQ(out, tf.make({1, 1, 3, 3}, {1, 2, 1, 2, 4, 2, 1, 2, 1}));
}

#ifndef USE_ATEN_LIB
TEST_F(OpCol2ImOutTest, HalfAccumulatesInFloat) {
  TensorFactory<ScalarType::Half> tf;
  std::vector<Half> in_data(81, Half(0.1f));
  for (size_t i = 36; i < 45; ++i) {
    in_data[i] = Half(1000.0f);
  }
  auto in = tf.make({1, 9, 9}, in_data);
  auto out = tf.zeros({1, 1, 3, 3});
  col2im(in, {3, 3}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, out);
  EXPECT_TENSOR_EQ(
      out,
      tf.make(
          {1, 1, 3, 3},
          {1000.5,
           1000.5,
           1000.5,
           1000.5,
           1001,
           1000.5,
           1000.5,
           1000.5,
           1000.5}));
}

TEST_F(OpCol2ImOutTest, NonDefaultDimOrder) {
  TensorFactory<ScalarType::Float> tf;
  auto in =
      tf.make_with_dimorder({2, 4, 4}, std::vector<float>(32, 1), {0, 2, 1});
  auto out = tf.zeros({2, 1, 3, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, out));
}
#endif

TEST_F(OpCol2ImOutTest, MismatchedBlockCount) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 4, 3});
  auto out = tf.zeros({1, 1, 3, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, out));
}

TEST_F(OpCol2ImOutTest, ChannelsNotDivisibleByKernel) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 3, 4});
  auto out = tf.zeros({1, 1, 3, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, out));
}

TEST_F(OpCol2ImOutTest, NoSlidingBlocks) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 9, 1});
  auto out = tf.zeros({1, 1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {1, 1}, {3, 3}, {1, 1}, {0, 0}, {1, 1}, out));
}

TEST_F(OpCol2ImOutTest, ZeroKernel) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 4, 4});
  auto out = tf.zeros({1, 1, 3, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3}, {0, 2}, {1, 1}, {0, 0}, {1, 1}, out));
}

TEST_F(OpCol2ImOutTest, ZeroStride) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 4, 4});
  auto out = tf.zeros({1, 1, 3, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3}, {2, 2}, {1, 1}, {0, 0}, {0, 1}, out));
}

TEST_F(OpCol2ImOutTest, ZeroDilation) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 4, 4});
  auto out = tf.zeros({1, 1, 3, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3}, {2, 2}, {0, 1}, {0, 0}, {1, 1}, out));
}

TEST_F(OpCol2ImOutTest, NegativePadding) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 4, 4});
  auto out = tf.zeros({1, 1, 3, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3}, {2, 2}, {1, 1}, {-1, 0}, {1, 1}, out));
}

TEST_F(OpCol2ImOutTest, ZeroOutputSize) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 4});
  auto out = tf.zeros({1, 1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {0, 0}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, out));
}

TEST_F(OpCol2ImOutTest, WrongParameterLength) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 4, 4});
  auto out = tf.zeros({1, 1, 3, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3}, {2}, {1, 1}, {0, 0}, {1, 1}, out));
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, out));
}

TEST_F(OpCol2ImOutTest, WrongInputRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 4, 4});
  auto out = tf.zeros({1, 1, 3, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, out));
}

TEST_F(OpCol2ImOutTest, EmptyChannels) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 0, 4});
  auto out = tf.zeros({1, 1, 3, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, out));
}

TEST_F(OpCol2ImOutTest, MismatchedDtype) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> other;
  auto in = tf.ones({1, 4, 4});
  auto out = other.zeros({1, 1, 3, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, out));
}

TEST_F(OpCol2ImOutTest, UnsupportedDtype) {
  TensorFactory<ScalarType::Long> tf;
  auto in = tf.ones({1, 4, 4});
  auto out = tf.zeros({1, 1, 3, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, col2im(in, {3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, out));
}
