# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.qualcomm.quantizer.qconfig import (
    get_ptq_per_block_quant_config,
    get_ptq_per_channel_quant_config,
    get_qat_per_channel_quant_config,
)
from executorch.backends.qualcomm.utils.constants import (
    DEFAULT_EPS_8BIT,
    DEFAULT_EPS_16BIT,
)


def _observer_eps(spec):
    """The eps of the observer a QuantizationSpec ends up constructing."""
    keywords = spec.observer_or_fake_quant_ctr.p.keywords
    if "observer" in keywords:
        # A fake-quant wrapper carries the real observer under "observer".
        keywords = keywords["observer"].p.keywords
    return keywords["eps"]


class WeightEpsTest(unittest.TestCase):
    """The per-channel weight scale floor has to match the weight width. Flooring
    an 8-bit scale at the 16-bit eps leaves a nearly-zero channel with a scale
    orders of magnitude below its neighbours, which the HTP computes as garbage.
    See https://github.com/pytorch/executorch/issues/23160.
    """

    def test_ptq_per_channel_weight_eps_follows_weight_dtype(self):
        for act_dtype in (torch.uint8, torch.uint16):
            with self.subTest(act_dtype=act_dtype):
                config = get_ptq_per_channel_quant_config(
                    act_dtype=act_dtype, weight_dtype=torch.int8
                )
                self.assertEqual(_observer_eps(config.weight), DEFAULT_EPS_8BIT)

                config = get_ptq_per_channel_quant_config(
                    act_dtype=act_dtype, weight_dtype=torch.int16
                )
                self.assertEqual(_observer_eps(config.weight), DEFAULT_EPS_16BIT)

    def test_qat_per_channel_weight_eps_follows_weight_dtype(self):
        config = get_qat_per_channel_quant_config(
            act_dtype=torch.uint16, weight_dtype=torch.int8
        )
        self.assertEqual(_observer_eps(config.weight), DEFAULT_EPS_8BIT)

        config = get_qat_per_channel_quant_config(
            act_dtype=torch.uint16, weight_dtype=torch.int16
        )
        self.assertEqual(_observer_eps(config.weight), DEFAULT_EPS_16BIT)

    def test_per_block_weight_eps_follows_weight_dtype(self):
        config = get_ptq_per_block_quant_config(
            act_dtype=torch.uint16, weight_dtype=torch.int8
        )
        self.assertEqual(_observer_eps(config.weight), DEFAULT_EPS_8BIT)

    def test_activation_eps_is_unchanged(self):
        config = get_ptq_per_channel_quant_config(
            act_dtype=torch.uint16, weight_dtype=torch.int8
        )
        self.assertEqual(_observer_eps(config.input_activation), DEFAULT_EPS_16BIT)

    def test_explicit_eps_still_wins_for_both(self):
        config = get_ptq_per_channel_quant_config(
            act_dtype=torch.uint16, weight_dtype=torch.int8, eps=1e-3
        )
        self.assertEqual(_observer_eps(config.weight), 1e-3)
        self.assertEqual(_observer_eps(config.input_activation), 1e-3)

        config = get_ptq_per_block_quant_config(
            act_dtype=torch.uint16, weight_dtype=torch.int8, eps=1e-3
        )
        self.assertEqual(_observer_eps(config.weight), 1e-3)
        self.assertEqual(_observer_eps(config.input_activation), 1e-3)


if __name__ == "__main__":
    unittest.main()
