/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core
#define PRECISION ${PRECISION}
layout(std430) buffer;
#include "common.glslh"

${layout_declare_tensor(B, "w", "t_quantized", "int", "buffer")}
${layout_declare_tensor(B, "w", "t_qparams", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_input", "float", "buffer", is_scalar_array=False)}
${layout_declare_ubo(B, "ivec4", "input_sizes")}
${layout_declare_ubo(B, "int", "input_numel")}

layout(push_constant) uniform Params { int qmin; int qmax; };
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
shared float row_min[64];
shared float row_max[64];

void main() {
  const int row = int(gl_WorkGroupID.y);
  const int lane = int(gl_LocalInvocationID.x);
  const int K4 = input_sizes.x / 4;
  float lo = 0.0;
  float hi = 0.0;
  for (int k = lane; k < K4; k += 64) {
    vec4 v = t_input[row * K4 + k];
    lo = min(lo, min(min(v.x, v.y), min(v.z, v.w)));
    hi = max(hi, max(max(v.x, v.y), max(v.z, v.w)));
  }
  row_min[lane] = lo;
  row_max[lane] = hi;
  barrier();
  for (int stride = 32; stride > 0; stride /= 2) {
    if (lane < stride) {
      row_min[lane] = min(row_min[lane], row_min[lane + stride]);
      row_max[lane] = max(row_max[lane], row_max[lane + stride]);
    }
    barrier();
  }
  const float scale = max((row_max[0] - row_min[0]) / float(qmax - qmin), 1.1920928955078125e-7);
  const int zp = int(clamp(float(qmin) - roundEven(row_min[0] / scale), float(qmin), float(qmax)));
  if (lane == 0) {
    t_qparams[2 * row] = scale;
    t_qparams[2 * row + 1] = float(zp);
  }
  for (int k = lane; k < K4; k += 64) {
    vec4 v = t_input[row * K4 + k];
    ivec4 q = ivec4(clamp(roundEven(v * (1.0 / scale)) + float(zp), float(qmin), float(qmax)));
    t_quantized[row * K4 + k] = pack_into_int32(q);
  }
}
