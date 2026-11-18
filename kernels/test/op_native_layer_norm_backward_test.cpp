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
#include <executorch/kernels/test/supported_features_skip.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ArrayRef;
using executorch::aten::IntArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::TensorFactory;

class OpNativeLayerNormBackwardOutTest : public OperatorTest {
 protected:
  std::tuple<Tensor&, Tensor&, Tensor&> op_native_layer_norm_backward_out(
      const Tensor& grad_out,
      const Tensor& input,
      const std::vector<int64_t>& normalized_shape,
      const Tensor& mean,
      const Tensor& rstd,
      const optional<Tensor>& weight,
      const optional<Tensor>& bias,
      std::array<bool, 3> output_mask_a,
      Tensor& grad_input,
      Tensor& grad_weight,
      Tensor& grad_bias) {
#ifndef USE_ATEN_LIB
    ArrayRef<bool> output_mask(output_mask_a.data(), output_mask_a.size());
#else
    std::array<bool, 3> output_mask = output_mask_a;
#endif
    return torch::executor::aten::native_layer_norm_backward_outf(
        context_,
        grad_out,
        input,
        IntArrayRef(normalized_shape.data(), normalized_shape.size()),
        mean,
        rstd,
        weight,
        bias,
        output_mask,
        grad_input,
        grad_weight,
        grad_bias);
  }

  // Values are exactly representable in every floating dtype. Expected
  // outputs come from torch.ops.aten.native_layer_norm_backward in float64.
  template <ScalarType DTYPE>
  void test_affine(double atol) {
    TensorFactory<DTYPE> tf;
    Tensor input = tf.make(
        {2, 2, 3}, {1, -0.5, 2, 0.25, -1.5, 0.75, 3, 1, -2, 0.5, 1.5, -0.25});
    Tensor grad_out = tf.make(
        {2, 2, 3}, {0.5, -1, 0.25, 2, -0.75, 1, -1.5, 0.5, 1, 0.25, -0.5, 2});
    Tensor weight = tf.make({2, 3}, {1.5, -0.5, 2, 0.25, 1, -1});
    Tensor bias = tf.make({2, 3}, {0.5, 0, -0.25, 1, 2, -1});
    Tensor mean = tf.make({2, 1, 1}, {0.5, 0.75});
    Tensor rstd = tf.make({2, 1, 1}, {2, 0.5});
    const std::vector<int64_t> normalized_shape = {2, 3};

    Tensor grad_input = tf.zeros({2, 2, 3});
    Tensor grad_weight = tf.zeros({2, 3});
    Tensor grad_bias = tf.zeros({2, 3});
    op_native_layer_norm_backward_out(
        grad_out,
        input,
        normalized_shape,
        mean,
        rstd,
        weight,
        bias,
        {true, true, true},
        grad_input,
        grad_weight,
        grad_bias);

    EXPECT_TENSOR_CLOSE_WITH_TOL(
        grad_input,
        tf.make(
            {2, 2, 3},
            {0.16666667,
             3.1666667,
             -2.6666667,
             1.4166667,
             3.0,
             -2.75,
             -0.45760091,
             0.16674805,
             0.72827148,
             0.22908529,
             0.13566081,
             -0.94303385}),
        1e-5,
        atol);
    EXPECT_TENSOR_CLOSE_WITH_TOL(
        grad_weight,
        tf.make({2, 3}, {-1.1875, 2.0625, -0.625, -1.03125, 2.8125, -0.5}),
        1e-5,
        atol);
    EXPECT_TENSOR_CLOSE_WITH_TOL(
        grad_bias,
        tf.make({2, 3}, {-1, -0.5, 1.25, 2.25, -1.25, 3}),
        1e-5,
        atol);
  }

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      torch::executor::TensorShapeDynamism dynamism) {
    TensorFactory<ScalarType::Float> tf;
    Tensor input = tf.make({3, 4}, {0, 1, 2, 3, -1, 4, 2, 0.5, 2, 2, -3, 1});
    Tensor grad_out =
        tf.make({3, 4}, {1, -1, 0.5, 2, 0.25, 0.5, -2, 1, -1, 3, 0.5, -0.5});
    Tensor weight = tf.make({4}, {1, 2, -1, 0.5});
    Tensor bias = tf.make({4}, {0, 1, -1, 2});
    Tensor mean = tf.make({3, 1}, {1.5, 1.375, 0.5});
    Tensor rstd = tf.make({3, 1}, {0.894423664, 0.540589154, 0.485070646});
    const std::vector<int64_t> normalized_shape = {4};

    Tensor grad_input = tf.zeros(out_shape, dynamism);
    Tensor grad_weight = tf.zeros({out_shape[1]}, dynamism);
    Tensor grad_bias = tf.zeros({out_shape[1]}, dynamism);
    op_native_layer_norm_backward_out(
        grad_out,
        input,
        normalized_shape,
        mean,
        rstd,
        weight,
        bias,
        {true, true, true},
        grad_input,
        grad_weight,
        grad_bias);

    // Matches torch autograd through torch.nn.functional.layer_norm.
    EXPECT_TENSOR_CLOSE(
        grad_input,
        tf.make(
            {3, 4},
            {1.2074704,
             -1.6099631,
             -0.40249012,
             0.80498288,
             -0.10490959,
             -0.2610371,
             0.5041798,
             -0.13823312,
             -1.3910104,
             2.0044841,
             0.15336565,
             -0.7668393}));
    EXPECT_TENSOR_CLOSE(
        grad_weight,
        tf.make({4}, {-2.3902163, 3.339553, -1.3010042, 2.0889878}));
    EXPECT_TENSOR_CLOSE(grad_bias, tf.make({4}, {0.25, 2.5, -1, 2.5}));
  }
};

TEST_F(OpNativeLayerNormBackwardOutTest, FloatTensors) {
  test_affine<ScalarType::Float>(1e-5);
}

TEST_F(OpNativeLayerNormBackwardOutTest, DoubleTensors) {
  test_affine<ScalarType::Double>(1e-5);
}

TEST_F(OpNativeLayerNormBackwardOutTest, HalfTensors) {
  test_affine<ScalarType::Half>(5e-3);
}

TEST_F(OpNativeLayerNormBackwardOutTest, BFloat16Tensors) {
  test_affine<ScalarType::BFloat16>(2e-2);
}

TEST_F(OpNativeLayerNormBackwardOutTest, MatchesAutograd) {
  test_dynamic_shape({3, 4}, torch::executor::TensorShapeDynamism::STATIC);
}

TEST_F(OpNativeLayerNormBackwardOutTest, NoWeight) {
  TensorFactory<ScalarType::Float> tf;
  Tensor input = tf.make(
      {2, 2, 3}, {1, -0.5, 2, 0.25, -1.5, 0.75, 3, 1, -2, 0.5, 1.5, -0.25});
  Tensor grad_out = tf.make(
      {2, 2, 3}, {0.5, -1, 0.25, 2, -0.75, 1, -1.5, 0.5, 1, 0.25, -0.5, 2});
  Tensor bias = tf.zeros({2, 3});
  Tensor mean = tf.make({2, 1, 1}, {0.5, 0.75});
  Tensor rstd = tf.make({2, 1, 1}, {2, 0.5});
  const std::vector<int64_t> normalized_shape = {2, 3};

  Tensor grad_input = tf.zeros({2, 2, 3});
  Tensor grad_weight = tf.zeros({2, 3});
  Tensor grad_bias = tf.zeros({2, 3});

  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen requires undefined outputs for masked-off gradients");

  op_native_layer_norm_backward_out(
      grad_out,
      input,
      normalized_shape,
      mean,
      rstd,
      optional<Tensor>(),
      bias,
      {true, false, true},
      grad_input,
      grad_weight,
      grad_bias);

  EXPECT_TENSOR_CLOSE(
      grad_input,
      tf.make(
          {2, 2, 3},
          {-1.5833333,
           1.1666667,
           -5.9166667,
           4.2916667,
           5.5,
           0.375,
           -0.50032552,
           0.14811198,
           -0.12923177,
           -0.064778646,
           -0.2639974,
           0.67838542}));
  EXPECT_TENSOR_EQ(grad_weight, tf.zeros({2, 3}));
  EXPECT_TENSOR_CLOSE(
      grad_bias, tf.make({2, 3}, {-1, -0.5, 1.25, 2.25, -1.25, 3}));
}

TEST_F(OpNativeLayerNormBackwardOutTest, OutputMaskCombinations) {
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen requires undefined outputs for masked-off gradients");

  TensorFactory<ScalarType::Float> tf;
  Tensor input = tf.make({3, 4}, {0, 1, 2, 3, -1, 4, 2, 0.5, 2, 2, -3, 1});
  Tensor grad_out =
      tf.make({3, 4}, {1, -1, 0.5, 2, 0.25, 0.5, -2, 1, -1, 3, 0.5, -0.5});
  Tensor weight = tf.make({4}, {1, 2, -1, 0.5});
  Tensor bias = tf.make({4}, {0, 1, -1, 2});
  Tensor mean = tf.make({3, 1}, {1.5, 1.375, 0.5});
  Tensor rstd = tf.make({3, 1}, {0.894423664, 0.540589154, 0.485070646});
  const std::vector<int64_t> normalized_shape = {4};

  Tensor expected_grad_input = tf.make(
      {3, 4},
      {1.2074704,
       -1.6099631,
       -0.40249012,
       0.80498288,
       -0.10490959,
       -0.2610371,
       0.5041798,
       -0.13823312,
       -1.3910104,
       2.0044841,
       0.15336565,
       -0.7668393});
  Tensor expected_grad_weight =
      tf.make({4}, {-2.3902163, 3.339553, -1.3010042, 2.0889878});
  Tensor expected_grad_bias = tf.make({4}, {0.25, 2.5, -1, 2.5});

  for (int bits = 0; bits < 8; ++bits) {
    const std::array<bool, 3> mask = {
        (bits & 1) != 0, (bits & 2) != 0, (bits & 4) != 0};
    SCOPED_TRACE(
        ::testing::Message() << "mask=" << mask[0] << mask[1] << mask[2]);
    Tensor grad_input = tf.full({3, 4}, 7);
    Tensor grad_weight = tf.full({4}, 7);
    Tensor grad_bias = tf.full({4}, 7);
    op_native_layer_norm_backward_out(
        grad_out,
        input,
        normalized_shape,
        mean,
        rstd,
        weight,
        bias,
        mask,
        grad_input,
        grad_weight,
        grad_bias);
    EXPECT_TENSOR_CLOSE(
        grad_input, mask[0] ? expected_grad_input : tf.full({3, 4}, 7));
    EXPECT_TENSOR_CLOSE(
        grad_weight, mask[1] ? expected_grad_weight : tf.full({4}, 7));
    EXPECT_TENSOR_CLOSE(
        grad_bias, mask[2] ? expected_grad_bias : tf.full({4}, 7));
  }
}

TEST_F(OpNativeLayerNormBackwardOutTest, EmptyRows) {
  TensorFactory<ScalarType::Float> tf;
  Tensor input = tf.zeros({0, 3});
  Tensor grad_out = tf.zeros({0, 3});
  Tensor weight = tf.ones({3});
  Tensor bias = tf.ones({3});
  Tensor mean = tf.zeros({0, 1});
  Tensor rstd = tf.zeros({0, 1});
  const std::vector<int64_t> normalized_shape = {3};

  Tensor grad_input = tf.zeros({0, 3});
  Tensor grad_weight = tf.ones({3});
  Tensor grad_bias = tf.ones({3});
  op_native_layer_norm_backward_out(
      grad_out,
      input,
      normalized_shape,
      mean,
      rstd,
      weight,
      bias,
      {true, true, true},
      grad_input,
      grad_weight,
      grad_bias);

  EXPECT_EQ(grad_input.numel(), 0);
  EXPECT_TENSOR_EQ(grad_weight, tf.zeros({3}));
  EXPECT_TENSOR_EQ(grad_bias, tf.zeros({3}));
}

TEST_F(
    OpNativeLayerNormBackwardOutTest,
    DynamicShapeUpperBoundLargerThanExpected) {
  test_dynamic_shape(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpNativeLayerNormBackwardOutTest, DynamicShapeUnbound) {
  ET_SKIP_IF(
      !torch::executor::testing::SupportedFeatures::get()->output_resize,
      "Dynamic shape unbound not supported");
  test_dynamic_shape(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}

TEST_F(OpNativeLayerNormBackwardOutTest, IntTensorsDies) {
  TensorFactory<ScalarType::Int> tf;
  Tensor input = tf.ones({2, 3});
  Tensor grad_out = tf.ones({2, 3});
  Tensor mean = tf.zeros({2, 1});
  Tensor rstd = tf.ones({2, 1});
  Tensor weight = tf.ones({3});
  Tensor bias = tf.zeros({3});
  Tensor grad_input = tf.zeros({2, 3});
  Tensor grad_weight = tf.zeros({3});
  Tensor grad_bias = tf.zeros({3});
  const std::vector<int64_t> normalized_shape = {3};

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_native_layer_norm_backward_out(
          grad_out,
          input,
          normalized_shape,
          mean,
          rstd,
          weight,
          bias,
          {true, true, true},
          grad_input,
          grad_weight,
          grad_bias));
}

class OpNativeLayerNormBackwardInvalidArgsTest
    : public OpNativeLayerNormBackwardOutTest {
 protected:
  TensorFactory<ScalarType::Float> tf;
  Tensor input = tf.ones({2, 3});
  Tensor grad_out = tf.ones({2, 3});
  Tensor mean = tf.zeros({2, 1});
  Tensor rstd = tf.ones({2, 1});
  optional<Tensor> weight = tf.ones({3});
  optional<Tensor> bias = tf.zeros({3});
  Tensor grad_input = tf.zeros({2, 3});
  Tensor grad_weight = tf.zeros({3});
  Tensor grad_bias = tf.zeros({3});
  std::vector<int64_t> normalized_shape = {3};
  std::array<bool, 3> mask = {true, true, true};

  void expect_args_rejected() {
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_native_layer_norm_backward_out(
            grad_out,
            input,
            normalized_shape,
            mean,
            rstd,
            weight,
            bias,
            mask,
            grad_input,
            grad_weight,
            grad_bias));
  }
};

TEST_F(OpNativeLayerNormBackwardInvalidArgsTest, WrongNormalizedShapeDies) {
  normalized_shape = {2};
  expect_args_rejected();
}

TEST_F(OpNativeLayerNormBackwardInvalidArgsTest, WrongWeightShapeDies) {
  weight = tf.ones({2});
  expect_args_rejected();
}

TEST_F(OpNativeLayerNormBackwardInvalidArgsTest, MismatchedOutputDtypeDies) {
  TensorFactory<ScalarType::Double> tf_double;
  grad_input = tf_double.zeros({2, 3});
  expect_args_rejected();
}

TEST_F(OpNativeLayerNormBackwardInvalidArgsTest, MissingWeightDies) {
  weight = optional<Tensor>();
  expect_args_rejected();
}

TEST_F(OpNativeLayerNormBackwardInvalidArgsTest, MissingBiasDies) {
  bias = optional<Tensor>();
  expect_args_rejected();
}

TEST_F(OpNativeLayerNormBackwardInvalidArgsTest, WrongMeanNumelDies) {
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen does not validate mean size");
  mean = tf.zeros({3, 1});
  expect_args_rejected();
}

TEST_F(OpNativeLayerNormBackwardInvalidArgsTest, WrongGradOutShapeDies) {
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen only requires grad_out to have as many elements as input");
  grad_out = tf.ones({3, 2});
  expect_args_rejected();
}

TEST_F(OpNativeLayerNormBackwardInvalidArgsTest, OutputResizeFailureDies) {
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen resizes outputs");
  grad_input = tf.zeros({4, 3});
  expect_args_rejected();
}

TEST_F(OpNativeLayerNormBackwardInvalidArgsTest, NonDefaultDimOrderDies) {
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen kernel can handle non-default dim order");
  input = tf.channels_last_like(tf.ones({1, 3, 2, 2}));
  grad_out = tf.channels_last_like(tf.ones({1, 3, 2, 2}));
  grad_input = tf.zeros_channels_last({1, 3, 2, 2});
  mean = tf.zeros({1, 3, 2, 1});
  rstd = tf.ones({1, 3, 2, 1});
  weight = tf.ones({2});
  bias = tf.zeros({2});
  grad_weight = tf.zeros({2});
  grad_bias = tf.zeros({2});
  normalized_shape = {2};
  expect_args_rejected();
}
