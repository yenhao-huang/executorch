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
#include <optional>
#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::TensorShapeDynamism;
using executorch::runtime::testing::TensorFactory;

class OpAvgPool2DBackwardOutTest : public OperatorTest {
 protected:
  Tensor& grad(
      const Tensor& grad_output,
      const Tensor& input,
      std::initializer_list<int64_t> kernel,
      std::initializer_list<int64_t> stride,
      std::initializer_list<int64_t> padding,
      bool ceil_mode,
      bool include_pad,
      std::optional<int64_t> divisor_override,
      Tensor& grad_input) {
    return torch::executor::aten::avg_pool2d_backward_outf(
        context_,
        grad_output,
        input,
        {kernel.begin(), kernel.size()},
        {stride.begin(), stride.size()},
        {padding.begin(), padding.size()},
        ceil_mode,
        include_pad,
        divisor_override,
        grad_input);
  }

  template <ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;
    auto grad_output = tf.make({1, 1, 2, 2}, {1.5, -2.5, 3.25, 0.5});
    auto input = tf.zeros({1, 1, 3, 3});
    auto grad_input = tf.zeros({1, 1, 3, 3});
    grad(
        grad_output,
        input,
        {2, 2},
        {1, 1},
        {0, 0},
        false,
        true,
        {},
        grad_input);
    EXPECT_TENSOR_CLOSE(
        grad_input,
        tf.make(
            {1, 1, 3, 3},
            {0.375,
             -0.25,
             -0.625,
             1.1875,
             0.6875,
             -0.5,
             0.8125,
             0.9375,
             0.125}));
  }
};

TEST_F(OpAvgPool2DBackwardOutTest, Basic4D) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output =
      tf.make({1, 2, 2, 2}, {1.0, 5.0, -1.0, 9.0, -4.0, -9.0, -1.0, -6.0});
  auto input = tf.zeros({1, 2, 4, 4});
  auto grad_input = tf.zeros({1, 2, 4, 4});
  grad(grad_output, input, {2, 2}, {2, 2}, {0, 0}, false, true, {}, grad_input);
  EXPECT_TENSOR_CLOSE(
      grad_input,
      tf.make({1, 2, 4, 4}, {0.25,  0.25,  1.25,  1.25,  0.25,  0.25,  1.25,
                             1.25,  -0.25, -0.25, 2.25,  2.25,  -0.25, -0.25,
                             2.25,  2.25,  -1.0,  -1.0,  -2.25, -2.25, -1.0,
                             -1.0,  -2.25, -2.25, -0.25, -0.25, -1.5,  -1.5,
                             -0.25, -0.25, -1.5,  -1.5}));
}

TEST_F(OpAvgPool2DBackwardOutTest, Unbatched) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output =
      tf.make({2, 2, 2}, {9.0, -3.0, 1.0, 9.0, 0.0, 1.0, 5.0, -8.0});
  auto input = tf.zeros({2, 3, 3});
  auto grad_input = tf.zeros({2, 3, 3});
  grad(grad_output, input, {2, 2}, {1, 1}, {0, 0}, false, true, {}, grad_input);
  EXPECT_TENSOR_CLOSE(
      grad_input,
      tf.make(
          {2, 3, 3},
          {2.25,
           1.5,
           -0.75,
           2.5,
           4.0,
           1.5,
           0.25,
           2.5,
           2.25,
           0.0,
           0.25,
           0.25,
           1.25,
           -0.5,
           -1.75,
           1.25,
           -0.75,
           -2.0}));
}

TEST_F(OpAvgPool2DBackwardOutTest, DefaultStride) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.make({1, 1, 2, 2}, {5.0, -8.0, 1.0, 6.0});
  auto input = tf.zeros({1, 1, 4, 6});
  auto grad_input = tf.zeros({1, 1, 4, 6});
  grad(grad_output, input, {2, 3}, {}, {0, 0}, false, true, {}, grad_input);
  EXPECT_TENSOR_CLOSE(
      grad_input,
      tf.make(
          {1, 1, 4, 6},
          {0.8333333, 0.8333333, 0.8333333, -1.3333334, -1.3333334, -1.3333334,
           0.8333333, 0.8333333, 0.8333333, -1.3333334, -1.3333334, -1.3333334,
           0.1666667, 0.1666667, 0.1666667, 1.0,        1.0,        1.0,
           0.1666667, 0.1666667, 0.1666667, 1.0,        1.0,        1.0}));
}

TEST_F(OpAvgPool2DBackwardOutTest, OverlappingWindows) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output =
      tf.make({2, 1, 2, 2}, {-4.0, 2.0, 7.0, 4.0, -8.0, -4.0, 4.0, 2.0});
  auto input = tf.zeros({2, 1, 4, 5});
  auto grad_input = tf.zeros({2, 1, 4, 5});
  grad(grad_output, input, {3, 2}, {1, 2}, {0, 0}, false, true, {}, grad_input);
  EXPECT_TENSOR_CLOSE(
      grad_input,
      tf.make(
          {2, 1, 4, 5}, {-0.6666667, -0.6666667, 0.3333333,  0.3333333,  0.0,
                         0.5,        0.5,        1.0,        1.0,        0.0,
                         0.5,        0.5,        1.0,        1.0,        0.0,
                         1.1666666,  1.1666666,  0.6666667,  0.6666667,  0.0,
                         -1.3333334, -1.3333334, -0.6666667, -0.6666667, 0.0,
                         -0.6666667, -0.6666667, -0.3333333, -0.3333333, 0.0,
                         -0.6666667, -0.6666667, -0.3333333, -0.3333333, 0.0,
                         0.6666667,  0.6666667,  0.3333333,  0.3333333,  0.0}));
}

TEST_F(OpAvgPool2DBackwardOutTest, PaddingIncludePad) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.make({1, 1, 2, 2}, {-6.0, -2.0, 7.0, 6.0});
  auto input = tf.zeros({1, 1, 3, 4});
  auto grad_input = tf.zeros({1, 1, 3, 4});
  grad(grad_output, input, {3, 3}, {2, 2}, {1, 1}, false, true, {}, grad_input);
  EXPECT_TENSOR_CLOSE(
      grad_input,
      tf.make(
          {1, 1, 3, 4},
          {-0.6666667,
           -0.8888889,
           -0.2222222,
           -0.2222222,
           0.1111111,
           0.5555556,
           0.4444445,
           0.4444445,
           0.7777778,
           1.4444444,
           0.6666667,
           0.6666667}));
}

TEST_F(OpAvgPool2DBackwardOutTest, PaddingExcludePad) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.make({1, 1, 2, 2}, {-6.0, -2.0, 7.0, 6.0});
  auto input = tf.zeros({1, 1, 3, 4});
  auto grad_input = tf.zeros({1, 1, 3, 4});
  grad(
      grad_output, input, {3, 3}, {2, 2}, {1, 1}, false, false, {}, grad_input);
  EXPECT_TENSOR_CLOSE(
      grad_input,
      tf.make(
          {1, 1, 3, 4},
          {-1.5,
           -1.8333334,
           -0.3333333,
           -0.3333333,
           0.25,
           0.9166666,
           0.6666666,
           0.6666666,
           1.75,
           2.75,
           1.0,
           1.0}));
}

TEST_F(OpAvgPool2DBackwardOutTest, CeilMode) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output =
      tf.make({1, 1, 3, 3}, {-8.0, 1.0, 6.0, 5.0, 0.0, 8.0, 5.0, 1.0, 5.0});
  auto input = tf.zeros({1, 1, 5, 5});
  auto grad_input = tf.zeros({1, 1, 5, 5});
  grad(grad_output, input, {2, 2}, {2, 2}, {0, 0}, true, true, {}, grad_input);
  EXPECT_TENSOR_CLOSE(
      grad_input,
      tf.make(
          {1, 1, 5, 5}, {-2.0, -2.0, 0.25, 0.25, 3.0, -2.0, -2.0, 0.25, 0.25,
                         3.0,  1.25, 1.25, 0.0,  0.0, 4.0,  1.25, 1.25, 0.0,
                         0.0,  4.0,  2.5,  2.5,  0.5, 0.5,  5.0}));
}

TEST_F(OpAvgPool2DBackwardOutTest, CeilModePaddingExcludePad) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output =
      tf.make({1, 1, 3, 3}, {1.0, 9.0, 0.0, 1.0, 5.0, -8.0, 1.0, 6.0, 5.0});
  auto input = tf.zeros({1, 1, 5, 4});
  auto grad_input = tf.zeros({1, 1, 5, 4});
  grad(grad_output, input, {3, 2}, {2, 2}, {1, 1}, true, false, {}, grad_input);
  EXPECT_TENSOR_CLOSE(
      grad_input,
      tf.make({1, 1, 5, 4}, {0.5,       2.25,      2.25,      0.0,
                             0.8333334, 3.0833333, 3.0833333, -2.6666667,
                             0.3333333, 0.8333333, 0.8333333, -2.6666667,
                             0.8333334, 2.3333333, 2.3333333, -0.1666667,
                             0.5,       1.5,       1.5,       2.5}));
}

TEST_F(OpAvgPool2DBackwardOutTest, CeilModePaddingIncludePad) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output =
      tf.make({1, 1, 3, 3}, {1.0, 9.0, 0.0, 1.0, 5.0, -8.0, 1.0, 6.0, 5.0});
  auto input = tf.zeros({1, 1, 5, 4});
  auto grad_input = tf.zeros({1, 1, 5, 4});
  grad(grad_output, input, {3, 2}, {2, 2}, {1, 1}, true, true, {}, grad_input);
  EXPECT_TENSOR_CLOSE(
      grad_input,
      tf.make({1, 1, 5, 4}, {0.1666667, 1.5,       1.5,       0.0,
                             0.3333333, 2.3333333, 2.3333333, -1.3333334,
                             0.1666667, 0.8333333, 0.8333333, -1.3333334,
                             0.3333333, 1.8333333, 1.8333333, -0.5000001,
                             0.1666667, 1.0,       1.0,       0.8333333}));
}

TEST_F(OpAvgPool2DBackwardOutTest, DivisorOverride) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.make(
      {1, 1, 4, 4},
      {-8.0,
       2.0,
       9.0,
       -3.0,
       1.0,
       9.0,
       0.0,
       1.0,
       5.0,
       -8.0,
       1.0,
       6.0,
       5.0,
       0.0,
       8.0,
       5.0});
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tf.zeros({1, 1, 4, 4});
  grad(grad_output, input, {3, 3}, {1, 1}, {1, 1}, false, false, 5, grad_input);
  EXPECT_TENSOR_CLOSE(
      grad_input,
      tf.make(
          {1, 1, 4, 4},
          {0.8,
           2.6,
           3.6,
           1.4,
           0.2,
           2.2,
           3.4,
           2.8,
           2.4,
           4.2,
           4.4,
           4.2,
           0.4,
           2.2,
           2.4,
           4.0}));
}

TEST_F(OpAvgPool2DBackwardOutTest, SingleElementParams) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output =
      tf.make({1, 1, 5, 5}, {-8.0, 2.0, 9.0,  -3.0, 1.0,  9.0,  0.0, 1.0, 5.0,
                             -8.0, 1.0, 6.0,  5.0,  0.0,  8.0,  5.0, 1.0, 5.0,
                             -1.0, 9.0, -4.0, -9.0, -1.0, -6.0, -4.0});
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tf.zeros({1, 1, 4, 4});
  grad(grad_output, input, {2}, {1}, {1}, false, true, {}, grad_input);
  EXPECT_TENSOR_CLOSE(
      grad_input,
      tf.make(
          {1, 1, 4, 4},
          {0.75,
           3.0,
           3.0,
           -1.25,
           4.0,
           3.0,
           2.75,
           1.25,
           3.25,
           4.25,
           2.25,
           4.0,
           -1.75,
           -1.0,
           -0.75,
           -0.5}));
}

TEST_F(OpAvgPool2DBackwardOutTest, EmptyBatch) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.make({0, 2, 1, 1}, {});
  auto input = tf.make({0, 2, 2, 2}, {});
  auto grad_input = tf.make({0, 2, 2, 2}, {});
  grad(grad_output, input, {2, 2}, {}, {0, 0}, false, true, {}, grad_input);
  EXPECT_TENSOR_EQ(grad_input, tf.make({0, 2, 2, 2}, {}));
}

TEST_F(OpAvgPool2DBackwardOutTest, AllFloatingDtypes) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ScalarType::dtype>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAvgPool2DBackwardOutTest, LongTruncatesPerWindow) {
  TensorFactory<ScalarType::Long> tf;
  auto grad_output = tf.make({1, 1, 2, 2}, {7, -7, 5, -5});
  auto input = tf.zeros({1, 1, 3, 3});
  auto grad_input = tf.zeros({1, 1, 3, 3});
  grad(grad_output, input, {2, 2}, {1, 1}, {0, 0}, false, true, {}, grad_input);
  EXPECT_TENSOR_EQ(
      grad_input, tf.make({1, 1, 3, 3}, {1, 0, -1, 2, 0, -2, 1, 0, -1}));
}

TEST_F(OpAvgPool2DBackwardOutTest, OverwritesStaleGradInput) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.make({1, 1, 1, 1}, {4.0});
  auto input = tf.zeros({1, 1, 3, 3});
  auto grad_input = tf.full({1, 1, 3, 3}, 100.0);
  grad(grad_output, input, {2, 2}, {2, 2}, {0, 0}, false, true, {}, grad_input);
  EXPECT_TENSOR_EQ(
      grad_input,
      tf.make({1, 1, 3, 3}, {1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0}));
}

TEST_F(OpAvgPool2DBackwardOutTest, DynamicResize) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.full({1, 1, 2, 2}, 4.0);
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tf.zeros({2, 2, 5, 5}, TensorShapeDynamism::DYNAMIC_BOUND);
  grad(grad_output, input, {2, 2}, {}, {0, 0}, false, true, {}, grad_input);
  EXPECT_TENSOR_EQ(grad_input, tf.ones({1, 1, 4, 4}));
}

TEST_F(OpAvgPool2DBackwardOutTest, GradOutputWrongShape) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.ones({1, 1, 2, 3});
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tf.zeros({1, 1, 4, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output,
          input,
          {2, 2},
          {2, 2},
          {0, 0},
          false,
          true,
          {},
          grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, GradOutputWrongChannels) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.ones({3, 2, 2});
  auto input = tf.zeros({2, 4, 4});
  auto grad_input = tf.zeros({2, 4, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output, input, {2, 2}, {}, {0, 0}, false, true, {}, grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, GradOutputWrongRank) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.ones({1, 2, 2});
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tf.zeros({1, 1, 4, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output, input, {2, 2}, {}, {0, 0}, false, true, {}, grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, MismatchedGradOutputDtype) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> tfDouble;
  auto grad_output = tfDouble.ones({1, 1, 2, 2});
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tf.zeros({1, 1, 4, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output, input, {2, 2}, {}, {0, 0}, false, true, {}, grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, MismatchedGradInputDtype) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> tfDouble;
  auto grad_output = tf.ones({1, 1, 2, 2});
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tfDouble.zeros({1, 1, 4, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output, input, {2, 2}, {}, {0, 0}, false, true, {}, grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, UnsupportedDtype) {
  TensorFactory<ScalarType::Int> tf;
  auto grad_output = tf.ones({1, 1, 2, 2});
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tf.zeros({1, 1, 4, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output, input, {2, 2}, {}, {0, 0}, false, true, {}, grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, ZeroKernel) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.ones({1, 1, 2, 2});
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tf.zeros({1, 1, 4, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output,
          input,
          {0, 2},
          {2, 2},
          {0, 0},
          false,
          true,
          {},
          grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, ZeroStride) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.ones({1, 1, 2, 2});
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tf.zeros({1, 1, 4, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output,
          input,
          {2, 2},
          {2, 0},
          {0, 0},
          false,
          true,
          {},
          grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, WrongStrideRank) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.ones({1, 1, 2, 2});
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tf.zeros({1, 1, 4, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output,
          input,
          {2, 2},
          {2, 2, 2},
          {0, 0},
          false,
          true,
          {},
          grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, ExcessPadding) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.ones({1, 1, 4, 4});
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tf.zeros({1, 1, 4, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output,
          input,
          {2, 2},
          {2, 2},
          {2, 2},
          false,
          true,
          {},
          grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, ZeroDivisorOverride) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.ones({1, 1, 2, 2});
  auto input = tf.zeros({1, 1, 4, 4});
  auto grad_input = tf.zeros({1, 1, 4, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(grad_output, input, {2, 2}, {}, {0, 0}, false, true, 0, grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, WrongInputRank) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.ones({2, 2});
  auto input = tf.zeros({4, 4});
  auto grad_input = tf.zeros({4, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output, input, {2, 2}, {}, {0, 0}, false, true, {}, grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, KernelLargerThanInput) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.ones({1, 1, 1});
  auto input = tf.zeros({1, 1, 1});
  auto grad_input = tf.zeros({1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output,
          input,
          {3, 3},
          {3, 3},
          {0, 0},
          false,
          true,
          {},
          grad_input));
}

TEST_F(OpAvgPool2DBackwardOutTest, CeilKernelLargerThanInput) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.full({1, 1, 1}, 5.0);
  auto input = tf.zeros({1, 1, 1});
  auto grad_input = tf.zeros({1, 1, 1});
  grad(grad_output, input, {3, 3}, {3, 3}, {0, 0}, true, true, {}, grad_input);
  EXPECT_TENSOR_EQ(grad_input, tf.full({1, 1, 1}, 5.0));
}

#ifndef USE_ATEN_LIB
TEST_F(OpAvgPool2DBackwardOutTest, HalfAccumulatesInFloat) {
  TensorFactory<ScalarType::Half> tf;
  // Half rounds 2048 + 1 back to 2048, so a Half running sum would never move
  // past 2048.
  auto grad_output = tf.make({1, 1, 1, 4}, {2048, 1, 1, 1});
  auto input = tf.zeros({1, 1, 1, 7});
  auto grad_input = tf.zeros({1, 1, 1, 7});
  grad(grad_output, input, {1, 4}, {1, 1}, {0, 0}, false, true, 1, grad_input);
  EXPECT_TENSOR_EQ(
      grad_input, tf.make({1, 1, 1, 7}, {2048, 2048, 2050, 2052, 3, 2, 1}));
}

TEST_F(OpAvgPool2DBackwardOutTest, NonDefaultDimOrder) {
  TensorFactory<ScalarType::Float> tf;
  auto grad_output = tf.ones({1, 2, 1, 1});
  auto input = tf.make_with_dimorder(
      {1, 2, 2, 2}, std::vector<float>(8, 1), {0, 2, 3, 1});
  auto grad_input = tf.make_with_dimorder(
      {1, 2, 2, 2}, std::vector<float>(8, 0), {0, 2, 3, 1});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      grad(
          grad_output, input, {2, 2}, {}, {0, 0}, false, true, {}, grad_input));
}
#endif
