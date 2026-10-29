/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core
${define_required_extensions("buffer", DTYPE)}
#define T ${texel_load_component_type(DTYPE, "buffer")}
#define PRECISION ${PRECISION}
$if USE_DOT:
  #extension GL_EXT_integer_dot_product : require
layout(std430) buffer;
#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_input", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_qparams", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_sums", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_scales", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}
${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "int", "output_numel")}
${layout_declare_spec_const(C, "int", "has_bias", "0")}
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int n4 = int(gl_GlobalInvocationID.x);
  const int N = output_sizes.x;
  const int K4 = input_sizes.x / 4;
  const int N4 = div_up_4(N);
  if (n4 >= N4) { return; }
  const int row_step = int(gl_NumWorkGroups.y * gl_WorkGroupSize.y);
  for (int row = int(gl_GlobalInvocationID.y); row < output_numel / N; row += row_step) {
    ivec4 accum = ivec4(0);
    for (int k4 = 0; k4 < K4; k4++) {
      const int a = t_input[row * K4 + k4];
      $if WEIGHT_STORAGE == "buffer":
        ivec4 w = t_weight[k4 * N4 + n4];
      $else:
        ivec4 w = texelFetch(t_weight, ivec2(n4, k4), 0);
      for (int i = 0; i < 4; i++) {
        $if USE_DOT:
          accum[i] = dotPacked4x8AccSatEXT(a, w[i], accum[i]);
        $else:
          ivec4 av = unpack_int8x4(a);
          ivec4 wv = unpack_int8x4(w[i]);
          accum[i] += av.x * wv.x + av.y * wv.y + av.z * wv.z + av.w * wv.w;
      }
    }
    const float scale = t_qparams[2 * row];
    const int zp = int(t_qparams[2 * row + 1]);
    for (int i = 0; i < 4; i++) {
      const int n = n4 * 4 + i;
      if (n < N) {
        float v = float(accum[i] - zp * t_sums[n]) * scale * float(t_scales[n]);
        if (has_bias != 0) { v += float(t_bias[n]); }
        t_output[row * N + n] = T(v);
      }
    }
  }
}
