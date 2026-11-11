/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::testing::TensorFactory;
using torch::executor::native::utils::SupportedTensorDtypes;
namespace utils = torch::executor::native::utils;

TEST(ElementwiseUtilTest, BinaryChannelsLastBroadcast) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make_channels_last({1, 2, 1, 2}, {1, 3, 2, 4});
  Tensor b = tf.make_channels_last({1, 2, 1, 1}, {10, 20});
  Tensor out = tf.make_channels_last({1, 2, 1, 2}, {0, 0, 0, 0});
  torch::executor::apply_binary_elementwise_fn<int32_t, int32_t, int32_t>(
      [](int32_t x, int32_t y) { return x * y; }, a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make_channels_last({1, 2, 1, 2}, {10, 60, 20, 80}));
}

TEST(ElementwiseUtilTest, TernaryChannelsLastBroadcast) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make_channels_last({1, 2, 1, 2}, {1, 3, 2, 4});
  Tensor b = tf.make_channels_last({1, 2, 1, 1}, {10, 20});
  Tensor c = tf.make({1}, {5});
  Tensor out = tf.make_channels_last({1, 2, 1, 2}, {0, 0, 0, 0});
  torch::executor::
      apply_ternary_elementwise_fn<int32_t, int32_t, int32_t, int32_t>(
          [](int32_t x, int32_t y, int32_t z) { return x * y + z; },
          a,
          b,
          c,
          out);
  EXPECT_TENSOR_EQ(out, tf.make_channels_last({1, 2, 1, 2}, {15, 65, 25, 85}));
}

template <ScalarType InputType>
void test_channels_last_elementwise_helpers() {
  TensorFactory<InputType> tf_a;
  TensorFactory<ScalarType::Float> tf;
  Tensor a = tf_a.make_channels_last({1, 2, 1, 2}, {1, 3, 2, 4});
  Tensor b = tf.make_channels_last({1, 2, 1, 1}, {10, 20});
  Tensor c = tf.make({1}, {5});
  Tensor out = tf.make_channels_last({1, 2, 1, 2}, {0, 0, 0, 0});
  KernelRuntimeContext ctx;
  ET_DEFINE_OPERATOR_NAME(op_name, "mul.out");
  utils::apply_bitensor_elementwise_fn<
      float,
      op_name,
      SupportedTensorDtypes::SAME_AS_COMMON>(
      [](float x, float y) { return x * y; },
      ctx,
      a,
      SupportedTensorDtypes::REALHBBF16,
      b,
      SupportedTensorDtypes::REALHBBF16,
      out);
  EXPECT_EQ(ctx.failure_state(), executorch::runtime::Error::Ok);
  EXPECT_TENSOR_EQ(out, tf.make_channels_last({1, 2, 1, 2}, {10, 60, 20, 80}));

  utils::apply_tritensor_elementwise_fn<
      float,
      op_name,
      SupportedTensorDtypes::SAME_AS_COMMON>(
      [](float x, float y, float z) { return x * y + z; },
      ctx,
      a,
      SupportedTensorDtypes::REALHBBF16,
      b,
      SupportedTensorDtypes::REALHBBF16,
      c,
      SupportedTensorDtypes::REALHBBF16,
      out);
  EXPECT_EQ(ctx.failure_state(), executorch::runtime::Error::Ok);
  EXPECT_TENSOR_EQ(out, tf.make_channels_last({1, 2, 1, 2}, {15, 65, 25, 85}));

  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor thresholds = tf.make_channels_last({1, 2, 1, 1}, {2, 4});
  Tensor comparison = tf_bool.make_channels_last({1, 2, 1, 2}, {0, 0, 0, 0});
  ET_DEFINE_OPERATOR_NAME(compare_op_name, "lt.Tensor_out");
  utils::apply_bitensor_elementwise_fn<
      float,
      compare_op_name,
      SupportedTensorDtypes::BOOL>(
      [](float x, float y) { return x < y; },
      ctx,
      a,
      SupportedTensorDtypes::REALHBBF16,
      thresholds,
      SupportedTensorDtypes::REALHBBF16,
      comparison);
  EXPECT_TENSOR_EQ(
      comparison, tf_bool.make_channels_last({1, 2, 1, 2}, {1, 1, 0, 0}));
}

TEST(ElementwiseUtilTest, SpecializedChannelsLastBroadcast) {
  test_channels_last_elementwise_helpers<ScalarType::Float>();
}

TEST(ElementwiseUtilTest, DtypeConvertingChannelsLastBroadcast) {
  test_channels_last_elementwise_helpers<ScalarType::Int>();
}

TEST(ElementwiseUtilTest, VectorizableSameShapeDifferentLayouts) {
  TensorFactory<ScalarType::Float> tf;
  Tensor a = tf.make({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  Tensor b =
      tf.make_channels_last({1, 2, 2, 2}, {10, 50, 20, 60, 30, 70, 40, 80});
  Tensor out = tf.make_channels_last({1, 2, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0});
  KernelRuntimeContext ctx;
  ET_DEFINE_OPERATOR_NAME(op_name, "add.out");
  utils::apply_bitensor_elementwise_fn<
      float,
      op_name,
      SupportedTensorDtypes::SAME_AS_COMMON>(
      [](auto x, auto y) { return x + y; },
      ctx,
      a,
      SupportedTensorDtypes::REALHBBF16,
      b,
      SupportedTensorDtypes::REALHBBF16,
      out);
  EXPECT_EQ(ctx.failure_state(), executorch::runtime::Error::Ok);
  EXPECT_TENSOR_EQ(
      out,
      tf.make_channels_last({1, 2, 2, 2}, {11, 55, 22, 66, 33, 77, 44, 88}));
}

TEST(ElementwiseUtilTest, ChannelsLastBroadcastAcrossChunks) {
  TensorFactory<ScalarType::Float> tf;
  constexpr int32_t width = 32771;
  std::vector<float> values(2 * width);
  std::vector<float> expected(2 * width);
  for (int32_t i = 0; i < 2 * width; ++i) {
    values[i] = i % 17;
    expected[i] = values[i] * (i % 2 == 0 ? 2 : 3);
  }
  Tensor a = tf.make_channels_last({1, 2, 1, width}, values);
  Tensor b = tf.make_channels_last({1, 2, 1, 1}, {2, 3});
  Tensor out =
      tf.make_channels_last({1, 2, 1, width}, std::vector<float>(2 * width));
  KernelRuntimeContext ctx;
  ET_DEFINE_OPERATOR_NAME(op_name, "mul.out");
  utils::apply_bitensor_elementwise_fn<
      float,
      op_name,
      SupportedTensorDtypes::SAME_AS_COMMON>(
      [](float x, float y) { return x * y; },
      ctx,
      a,
      SupportedTensorDtypes::REALHBBF16,
      b,
      SupportedTensorDtypes::REALHBBF16,
      out);
  EXPECT_TENSOR_EQ(out, tf.make_channels_last({1, 2, 1, width}, expected));
}

TEST(ElementwiseUtilTest, UnaryDifferentLayouts) {
  TensorFactory<ScalarType::Float> tf;
  Tensor a = tf.make({1, 2, 1, 2}, {1, 2, 3, 4});
  Tensor out = tf.make_channels_last({1, 2, 1, 2}, {0, 0, 0, 0});
  KernelRuntimeContext ctx;
  ET_DEFINE_OPERATOR_NAME(op_name, "neg.out");
  utils::apply_unitensor_elementwise_fn<
      float,
      op_name,
      SupportedTensorDtypes::SAME_AS_COMMON>(
      [](auto x) { return -x; },
      ctx,
      a,
      SupportedTensorDtypes::REALHBBF16,
      out);
  EXPECT_TENSOR_EQ(out, tf.make_channels_last({1, 2, 1, 2}, {-1, -3, -2, -4}));
}

TEST(ElementwiseUtilTest, ScalarAndEmpty) {
  TensorFactory<ScalarType::Float> tf;
  KernelRuntimeContext ctx;
  ET_DEFINE_OPERATOR_NAME(op_name, "mul.out");
  for (const auto& shape : {std::vector<int32_t>{}, std::vector<int32_t>{0}}) {
    Tensor a = tf.full(shape, 2);
    Tensor b = tf.full(shape, 3);
    Tensor out = tf.zeros(shape);
    utils::apply_bitensor_elementwise_fn<
        float,
        op_name,
        SupportedTensorDtypes::SAME_AS_COMMON>(
        [](float x, float y) { return x * y; },
        ctx,
        a,
        SupportedTensorDtypes::REALHBBF16,
        b,
        SupportedTensorDtypes::REALHBBF16,
        out);
    EXPECT_TENSOR_EQ(out, tf.full(shape, 6));
  }
}
