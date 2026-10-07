/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void scalar_tensor(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef scalar_in = args[0];

  // Get the output tensor reference
  ValueRef out = args[args.size() - 1];

  // The shader declares its scalar parameter with the type named by the last
  // dtype suffix, so the parameter buffer must carry that exact type. Uploading
  // float bytes to an int declaration reinterprets the bit pattern (3.0f reads
  // back as 1077936128), and routing an integer through float loses values
  // above 2**24. Bool and Int scalars therefore both travel as int32.
  const vkapi::ScalarType scalar_dtype = graph.dtype_of(scalar_in);
  const bool scalar_is_integral =
      scalar_dtype == vkapi::kBool || scalar_dtype == vkapi::kInt;

  std::string kernel_name("scalar_tensor");
  kernel_name.reserve(kShaderNameReserve);

  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(
      kernel_name, scalar_is_integral ? vkapi::kInt : vkapi::kFloat);

  vkapi::ParamsBindList scalar_param;
  if (scalar_is_integral) {
    scalar_param = {
        graph.create_params_buffer(graph.extract_scalar<int32_t>(scalar_in))};
  } else {
    scalar_param = {
        graph.create_params_buffer(graph.extract_scalar<float>(scalar_in))};
  }

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_gwg(out),
      graph.create_lwg(out),
      // Inputs and Outputs
      {{out, vkapi::kWrite}},
      // Shader params buffers
      scalar_param,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.scalar_tensor.default, scalar_tensor);
}

} // namespace vkcompute
