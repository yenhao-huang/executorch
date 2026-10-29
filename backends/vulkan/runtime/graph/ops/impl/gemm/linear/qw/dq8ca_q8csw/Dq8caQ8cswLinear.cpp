/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {
namespace {

void resize_token_quantize(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>&) {
  const auto input = args.at(1).refs.at(0);
  const int64_t K = graph->size_at<int64_t>(-1, input);
  const int64_t M = graph->numel_of(input) / K;
  graph->virtual_resize(args.at(0).refs.at(0), {M, K / 4});
  graph->virtual_resize(args.at(0).refs.at(1), {M, 2});
}

GlobalWorkGrid token_quantize_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo&,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>&) {
  const auto input = args.at(1).refs.at(0);
  const uint32_t rows =
      graph->numel_of(input) / graph->size_at<int32_t>(-1, input);
  const auto max_groups =
      graph->context()->adapter_ptr()->max_compute_workgroup_count();
  return GlobalWorkGrid(
      {64u, std::min(rows, max_groups[1]), 1u},
      kTiledWorkGrid,
      LocalWorkGroup(64u, 1u, 1u));
}

void resize_token_linear(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  auto sizes = graph->sizes_of(resize_args.at(0));
  sizes.back() = graph->size_at<int64_t>(-2, resize_args.at(1));
  graph->virtual_resize(args.at(0).refs.at(0), sizes);
}

GlobalWorkGrid token_linear_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo&,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>&) {
  const auto output = args.at(0).refs.at(0);
  const uint32_t N = graph->size_at<uint32_t>(-1, output);
  const uint32_t M = graph->numel_of(output) / N;
  const auto max_groups =
      graph->context()->adapter_ptr()->max_compute_workgroup_count();
  return GlobalWorkGrid(
      {utils::div_up_4(N), std::min(M, max_groups[1]), 1u}, kTiledWorkGrid);
}

void linear_dq8ca_q8csw(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  const auto input = args.at(0);
  const auto weights = args.at(1);
  const auto sums = args.at(2);
  const auto scales = args.at(3);
  const auto bias = args.at(4);
  const auto output = args.back();
  const int32_t qmin = graph.extract_scalar<int32_t>(args.at(5));
  const int32_t qmax = graph.extract_scalar<int32_t>(args.at(6));
  const int64_t K = graph.size_at<int64_t>(-1, input);
  VK_CHECK_COND(K > 0 && K % 4 == 0);
  const int64_t M = graph.numel_of(input) / K;
  VK_CHECK_COND(qmin >= -128 && qmin < qmax && qmax <= 127);
  const auto dtype = graph.dtype_of(input);
  VK_CHECK_COND(dtype == vkapi::kFloat || dtype == vkapi::kHalf);
  VK_CHECK_COND(
      graph.dtype_of(output) == dtype && graph.dtype_of(scales) == dtype);
  VK_CHECK_COND(graph.val_is_none(bias) || graph.dtype_of(bias) == dtype);
  VK_CHECK_COND(
      graph.is_buffer_storage(input) && graph.is_buffer_storage(output));

  const auto packed_weight = prepack_quantized_linear_weight(
      graph, QuantizationConfig(8, kPerChannel, {K}), weights);
  const auto packed_sums =
      prepack_standard(graph, sums, utils::kBuffer, utils::kWidthPacked);
  const auto packed_scales =
      prepack_standard(graph, scales, utils::kBuffer, utils::kWidthPacked);
  TmpTensor dummy_bias(&graph, {1}, dtype, utils::kBuffer, utils::kWidthPacked);
  const bool has_bias = graph.val_is_not_none(bias);
  const auto packed_bias = has_bias
      ? prepack_standard(graph, bias, utils::kBuffer, utils::kWidthPacked)
      : dummy_bias.vref;

  TmpTensor quantized(
      &graph, {M, K / 4}, vkapi::kInt, utils::kBuffer, utils::kWidthPacked);
  TmpTensor qparams(
      &graph, {M, 2}, vkapi::kFloat, utils::kBuffer, utils::kWidthPacked);
  std::string quantize_shader = "dq8ca_q8csw_quantize";
  add_dtype_suffix(quantize_shader, dtype);
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(quantize_shader),
      token_quantize_gwg,
      pick_required_lwg,
      {{{quantized.vref, qparams.vref}, vkapi::kWrite}, {input, vkapi::kRead}},
      {graph.sizes_ubo(input), graph.numel_ubo(input)},
      {PushConstantDataInfo(&qmin, sizeof(qmin)),
       PushConstantDataInfo(&qmax, sizeof(qmax))},
      {},
      {},
      resize_token_quantize));

  std::string shader = "linear_dq8ca_q8csw";
  add_storage_type_suffix(shader, graph.storage_type_of(packed_weight));
  shader += graph.can_use_int8_dot_product() ? "_dot" : "_nodot";
  add_dtype_suffix(shader, dtype);
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(shader),
      token_linear_gwg,
      pick_xy_square_lwg,
      {{output, vkapi::kWrite},
       {{quantized.vref,
         qparams.vref,
         packed_weight,
         packed_sums,
         packed_scales,
         packed_bias},
        vkapi::kRead}},
      {graph.sizes_ubo(input),
       graph.sizes_ubo(output),
       graph.numel_ubo(output)},
      {},
      {uint32_t(has_bias)},
      {input, weights},
      resize_token_linear));
}

} // namespace

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_dq8ca_q8csw.default, linear_dq8ca_q8csw);
}

} // namespace vkcompute
