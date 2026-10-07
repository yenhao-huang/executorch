# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch

from executorch.examples.models.llama.rope import (
    apply_rotary_emb,
    apply_rotary_emb_to_k,
    precompute_freqs_cis,
)


def reference_rope(k: torch.Tensor, head_dim: int, start_pos: int) -> torch.Tensor:
    """RoPE written out element by element in float64.

    This is deliberately independent of rope.py: it is the formula a user would
    reach for when reimplementing RoPE outside the model, as in
    https://github.com/pytorch/executorch/issues/5170, evaluated in double so
    it can serve as ground truth for either side.
    """
    out = torch.empty_like(k, dtype=torch.float64)
    batch, seq_len, n_heads, _ = k.shape
    for b in range(batch):
        for s in range(seq_len):
            position = start_pos + s
            for h in range(n_heads):
                for pair in range(head_dim // 2):
                    theta = 1.0 / math.pow(10000.0, (2.0 * pair / head_dim))
                    angle = position * theta
                    real = float(k[b, s, h, 2 * pair])
                    imag = float(k[b, s, h, 2 * pair + 1])
                    out[b, s, h, 2 * pair] = real * math.cos(angle) - imag * math.sin(
                        angle
                    )
                    out[b, s, h, 2 * pair + 1] = real * math.sin(
                        angle
                    ) + imag * math.cos(angle)
    return out


class RopeNumericsTest(unittest.TestCase):
    """The RoPE in rope.py computes the same rotation as the textbook formula;
    what differs from a float32 recomputation is only the dtype the result is
    cast back to. See https://github.com/pytorch/executorch/issues/5170.
    """

    head_dim = 16
    seq_len = 3
    n_heads = 2

    def _inputs(self, dtype: torch.dtype):
        torch.manual_seed(0)
        shape = (1, self.seq_len, self.n_heads, self.head_dim)
        return torch.randn(shape).to(dtype)

    def test_float32_matches_the_textbook_formula(self):
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, 8)
        k = self._inputs(torch.float32)

        got = apply_rotary_emb_to_k(k, freqs_cos[: self.seq_len], freqs_sin[: self.seq_len])
        expected = reference_rope(k.double(), self.head_dim, start_pos=0)

        self.assertEqual(got.dtype, torch.float32)
        torch.testing.assert_close(
            got.double(), expected, atol=1e-6, rtol=0, check_dtype=False
        )

    def test_apply_rotary_emb_rotates_q_and_k_the_same_way(self):
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, 8)
        q = self._inputs(torch.float32)
        k = self._inputs(torch.float32) * 2

        q_out, k_out = apply_rotary_emb(
            q, k, freqs_cos[: self.seq_len], freqs_sin[: self.seq_len]
        )
        torch.testing.assert_close(
            q_out.double(),
            reference_rope(q.double(), self.head_dim, 0),
            atol=1e-6,
            rtol=0,
            check_dtype=False,
        )
        torch.testing.assert_close(
            k_out.double(),
            reference_rope(k.double(), self.head_dim, 0),
            atol=1e-6,
            rtol=0,
            check_dtype=False,
        )

    def test_offset_position_matches(self):
        start_pos = 5
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, 16)
        k = self._inputs(torch.float32)

        got = apply_rotary_emb_to_k(
            k,
            freqs_cos[start_pos : start_pos + self.seq_len],
            freqs_sin[start_pos : start_pos + self.seq_len],
        )
        torch.testing.assert_close(
            got.double(),
            reference_rope(k.double(), self.head_dim, start_pos),
            atol=1e-6,
            rtol=0,
            check_dtype=False,
        )

    def test_reduced_precision_differs_only_by_the_output_dtype(self):
        """The result is cast back to the input dtype, so comparing it against a
        float32 recomputation shows that dtype's resolution, not a different
        rotation. This is what the issue reported as a mismatch.
        """
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, 8)
        for dtype, tolerance in (
            (torch.float32, 1e-6),
            (torch.float16, 1e-3),
            (torch.bfloat16, 1e-2),
        ):
            with self.subTest(dtype=dtype):
                k = self._inputs(dtype)
                got = apply_rotary_emb_to_k(
                    k, freqs_cos[: self.seq_len], freqs_sin[: self.seq_len]
                )
                self.assertEqual(got.dtype, dtype)
                error = (
                    (got.double() - reference_rope(k.double(), self.head_dim, 0))
                    .abs()
                    .max()
                    .item()
                )
                self.assertLess(error, tolerance)
                if dtype is not torch.float32:
                    # Large enough to be the dtype rather than float32 rounding.
                    self.assertGreater(error, 1e-5)


if __name__ == "__main__":
    unittest.main()
