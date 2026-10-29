/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include "test_utils.h"

#include <cassert>

class VulkanLinearQCS4WTest : public ::testing::Test {
 public:
  void SetUp() override {
    if (!vkcompute::api::context()
             ->adapter_ptr()
             ->supports_int16_shader_types()) {
      GTEST_SKIP();
    }
  }

  void TearDown() override {
    // Clean up any resources if needed
  }
};

//
// Reference Implementations
//

at::Tensor unpack_weights_4x2(const at::Tensor& weights_4x2) {
  std::vector<int64_t> weights_shape(weights_4x2.sizes().vec());
  weights_shape[1] *= 2;

  at::Tensor weights_unpacked =
      at::empty(weights_shape, at::device(at::kCPU).dtype(at::kInt));

  const int64_t N = weights_unpacked.size(0);
  const int64_t K = weights_unpacked.size(1);

  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k += 2) {
      const uint8_t packed_val = weights_4x2[n][k / 2].item().to<uint8_t>();
      const uint8_t second_val = packed_val & 0x0F;
      const uint8_t first_val = (packed_val & 0xF0) >> 4;

      weights_unpacked[n][k] = int(first_val);
      weights_unpacked[n][k + 1] = int(second_val);
    }
  }

  return weights_unpacked;
}

at::Tensor dequantize_and_linear_qcs4w(
    const at::Tensor& x,
    const at::Tensor& weights_4x2,
    const at::Tensor& scales) {
  std::vector<int64_t> weights_shape(weights_4x2.sizes().vec());
  weights_shape[1] *= 2;

  at::Tensor weights_dequantized =
      at::empty(weights_shape, at::device(at::kCPU).dtype(at::kFloat));

  const int64_t N = weights_dequantized.size(0);
  const int64_t K = weights_dequantized.size(1);

  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k += 2) {
      // const int scale_idx = k_groups * n + group_idx;
      const uint8_t packed_val = weights_4x2[n][k / 2].item().to<uint8_t>();
      const uint8_t second_val = packed_val & 0x0F;
      const uint8_t first_val = (packed_val & 0xF0) >> 4;

      const float scale = scales[n].item().to<float>();

      weights_dequantized[n][k] = (float(first_val) - 8.0) * scale;
      weights_dequantized[n][k + 1] = (float(second_val) - 8.0) * scale;
    }
  }

  return at::linear(x, weights_dequantized);
}

at::Tensor linear_qcs4w_reference_impl(
    const at::Tensor& x,
    const at::Tensor& weights_4x2,
    const at::Tensor& scales) {
  const std::vector<int64_t> original_x_size(x.sizes().vec());
  const size_t ndim = original_x_size.size();
  const int64_t out_features = weights_4x2.size(0);
  const at::Tensor x_flattened = x.reshape({-1, original_x_size[ndim - 1]});

  const at::Tensor weights_unpacked =
      (unpack_weights_4x2(weights_4x2) - 8).to(at::kChar);
  at::Tensor out =
      at::_weight_int8pack_mm(x_flattened, weights_unpacked, scales);

  std::vector<int64_t> out_shape(
      original_x_size.begin(), original_x_size.end());
  out_shape.at(ndim - 1) = out_features;
  return out.reshape(out_shape);
}

//
// Test functions
//

void test_reference_linear_qcs4w(
    const int B,
    const int M,
    const int K,
    const int N) {
  at::Tensor x = at::rand({B, M, K}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weights_4x2 =
      at::randint(0, 256, {N, K / 2}, at::device(at::kCPU).dtype(at::kByte));
  at::Tensor weights_int = unpack_weights_4x2(weights_4x2);

  at::Tensor scales = at::rand({N}, at::device(at::kCPU).dtype(at::kFloat));

  at::Tensor out = linear_qcs4w_reference_impl(x, weights_4x2, scales);

  at::Tensor out_ref = dequantize_and_linear_qcs4w(x, weights_4x2, scales);

  ASSERT_TRUE(at::allclose(out, out_ref));
}

void test_vulkan_linear_qcs4w_impl(
    const int B,
    const int M,
    const int K,
    const int N,
    const vkcompute::utils::StorageType in_storage =
        vkcompute::utils::kTexture3D,
    const vkcompute::utils::StorageType out_storage =
        vkcompute::utils::kTexture3D) {
  at::Tensor x = at::rand({B, M, K}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weights_4x2 =
      at::randint(0, 256, {N, K / 2}, at::device(at::kCPU).dtype(at::kByte));

  at::Tensor scales = at::rand({N}, at::device(at::kCPU).dtype(at::kFloat));

  at::Tensor out_ref = linear_qcs4w_reference_impl(x, weights_4x2, scales);

  // Build Vulkan graph
  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

#define MAKE_TENSORREF_FOR(x)              \
  ValueRef r_##x = graph.add_tensorref(    \
      x.sizes().vec(),                     \
      from_at_scalartype(x.scalar_type()), \
      x.const_data_ptr());

  MAKE_TENSORREF_FOR(weights_4x2);
  MAKE_TENSORREF_FOR(scales);

  IOValueRef r_x = graph.add_input_tensor(
      x.sizes().vec(), from_at_scalartype(x.scalar_type()), in_storage);

  const ValueRef r_out = graph.add_tensor(
      out_ref.sizes().vec(),
      from_at_scalartype(out_ref.scalar_type()),
      out_storage);

  VK_GET_OP_FN("et_vk.linear_qcs4w.default")
  (graph, {r_x.value, r_weights_4x2, r_scales, r_out});

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();

  graph.prepack();

  //
  // Run model
  //

  graph.propagate_resize();
  graph.maybe_cast_and_copy_into_staging(
      r_x.staging,
      x.const_data_ptr(),
      x.numel(),
      from_at_scalartype(x.scalar_type()));

  graph.execute();

  at::Tensor vk_out = at::empty_like(out_ref);
  graph.maybe_cast_and_copy_from_staging(
      staging_out,
      vk_out.mutable_data_ptr(),
      vk_out.numel(),
      from_at_scalartype(vk_out.scalar_type()));

  ASSERT_TRUE(at::allclose(vk_out, out_ref, 1e-4, 1e-4));
}

void test_vulkan_linear_qcs4w(
    const int B,
    const int M,
    const int K,
    const int N) {
  test_vulkan_linear_qcs4w_impl(
      B, M, K, N, vkcompute::utils::kBuffer, vkcompute::utils::kBuffer);

  test_vulkan_linear_qcs4w_impl(
      B, M, K, N, vkcompute::utils::kTexture3D, vkcompute::utils::kTexture3D);
}

// Test linear_qcs4w operator

TEST_F(VulkanLinearQCS4WTest, test_reference_impl) {
  test_reference_linear_qcs4w(
      /*B = */ 1,
      /*M = */ 4,
      /*K = */ 128,
      /*N = */ 32);
}

TEST_F(VulkanLinearQCS4WTest, test_vulkan_impl_small_m) {
  test_vulkan_linear_qcs4w(
      /*B = */ 1,
      /*M = */ 4,
      /*K = */ 128,
      /*N = */ 32);

  test_vulkan_linear_qcs4w(
      /*B = */ 1,
      /*M = */ 1,
      /*K = */ 256,
      /*N = */ 256);
}

TEST_F(VulkanLinearQCS4WTest, test_vulkan_impl_gemm) {
  test_vulkan_linear_qcs4w(
      /*B = */ 1,
      /*M = */ 32,
      /*K = */ 32,
      /*N = */ 32);
}

TEST(VulkanPerTokenInt8Test, IntegerDotAndFallback) {
  using namespace vkcompute;
  auto* adapter = api::context()->adapter_ptr();
  for (const bool half : {false, true}) {
    const auto dtype = half ? vkapi::kHalf : vkapi::kFloat;
    for (const bool disable_dot : {false, true}) {
      vkapi::AdapterCapabilityOverrides overrides;
      if (disable_dot) {
        overrides =
            vkapi::AdapterCapabilityOverrides::without_dot_product_support();
      }
      vkapi::ScopedAdapterCapabilityOverride override(adapter, overrides);
      const std::vector<int64_t> features{
          4, 64, 256, 4 * (int64_t(adapter->max_texture2d_dim()) + 1)};
      for (const int64_t K : features) {
        for (const bool with_bias : {false, true}) {
          SCOPED_TRACE(
              ::testing::Message()
              << "K=" << K << " bias=" << with_bias
              << " disable_dot=" << disable_dot << " half=" << half);
          at::manual_seed(22431);
          auto x = at::randn({2, 5, K});
          x[0][0].zero_();
          x[0][1].fill_(1e-8);
          x[1][4].mul_(100);
          if (half) {
            x = x.to(at::kHalf).to(at::kFloat);
          }
          auto weights = at::randint(-128, 128, {7, K}, at::kChar);
          auto sums = weights.to(at::kInt).sum(1).to(at::kInt);
          auto scales =
              (at::rand({7}) * 0.01).to(half ? at::kHalf : at::kFloat);
          auto bias = at::randn({7}).to(half ? at::kHalf : at::kFloat);
          auto lo = at::amin(x, {-1}, true).clamp_max(0);
          auto hi = at::amax(x, {-1}, true).clamp_min(0);
          auto scale = ((hi - lo) / 255).clamp_min(1.1920928955078125e-7);
          auto zero = (-128 - (lo / scale).round()).clamp(-128, 127);
          auto q = (x * scale.reciprocal()).round().add(zero).clamp(-128, 127);
          auto reference = at::linear(
              (q - zero) * scale,
              weights.to(at::kFloat) * scales.to(at::kFloat).unsqueeze(1));
          if (with_bias) {
            reference += bias.to(at::kFloat);
          }

          if (half) {
            reference = reference.to(at::kHalf).to(at::kFloat);
          }

          GraphConfig config;
          config.external_adapter = adapter;
          ComputeGraph graph(config);
          if (disable_dot) {
            ASSERT_FALSE(graph.can_use_int8_dot_product());
          }
          auto input = graph.add_input_tensor(
              x.sizes().vec(), dtype, utils::kBuffer, utils::kWidthPacked);
          auto output = graph.add_tensor(
              reference.sizes().vec(),
              dtype,
              utils::kBuffer,
              utils::kWidthPacked);
          auto weight_ref = graph.add_tensorref(
              weights.sizes().vec(), vkapi::kChar, weights.const_data_ptr());
          auto sum_ref = graph.add_tensorref(
              sums.sizes().vec(), vkapi::kInt, sums.const_data_ptr());
          auto scale_ref = graph.add_tensorref(
              scales.sizes().vec(), dtype, scales.const_data_ptr());
          auto bias_ref = with_bias
              ? graph.add_tensorref(
                    bias.sizes().vec(), dtype, bias.const_data_ptr())
              : graph.add_none();
          VK_GET_OP_FN("et_vk.linear_dq8ca_q8csw.default")
          (graph,
           {input.value,
            weight_ref,
            sum_ref,
            scale_ref,
            bias_ref,
            graph.add_scalar<int64_t>(-128),
            graph.add_scalar<int64_t>(127),
            output});
          auto staging = graph.set_output_tensor(output);
          graph.prepare();
          graph.prepack();
          graph.propagate_resize();
          auto stored_x = x.to(half ? at::kHalf : at::kFloat);
          graph.maybe_cast_and_copy_into_staging(
              input.staging,
              stored_x.const_data_ptr(),
              stored_x.numel(),
              dtype);
          graph.execute();
          auto actual =
              at::empty_like(reference).to(half ? at::kHalf : at::kFloat);
          graph.maybe_cast_and_copy_from_staging(
              staging, actual.mutable_data_ptr(), actual.numel(), dtype);
          ASSERT_TRUE(at::allclose(
              actual.to(at::kFloat),
              reference,
              half ? 1e-3 : 2e-4,
              half ? 1e-3 : 2e-4));
        }
      }
    }
  }
}
