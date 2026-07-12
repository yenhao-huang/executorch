# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch

from executorch.examples.models.llama.runner.generation import LlamaRunner


class _Tokenizer:
    eos_id = 3
    stop_tokens = [3, 4]
    n_words = 200064

    def decode_token(self, token):
        return f"<{token}>"


class _Runner(LlamaRunner):
    def __init__(self, output_tokens, *, eos_ids=None, use_kv_cache=True):
        self.output_tokens = iter(output_tokens)
        self.calls = []
        with patch(
            "executorch.examples.models.llama.runner.generation.get_tokenizer",
            return_value=_Tokenizer(),
        ):
            super().__init__(
                tokenizer_path="tokenizer.model",
                max_seq_len=16,
                max_batch_size=1,
                use_kv_cache=use_kv_cache,
                vocab_size=200064,
                eos_ids=eos_ids,
            )

    def forward(self, tokens, input_pos=None):
        self.calls.append(
            (
                tokens.tolist(),
                None if input_pos is None else input_pos.tolist(),
            )
        )
        output_token = next(self.output_tokens)
        logits = torch.full((1, 200064), -1.0)
        logits[0, output_token] = 1.0
        return logits


@pytest.mark.parametrize("terminal_id", [200020, 199999])
@pytest.mark.parametrize("use_kv_cache", [False, True])
def test_generation_stops_on_each_explicit_terminal_id(
    terminal_id, use_kv_cache, capsys
):
    runner = _Runner(
        [5, terminal_id, 7],
        eos_ids=[200020, 199999],
        use_kv_cache=use_kv_cache,
    )

    result = runner.generate([1, 2], max_seq_len=6, temperature=0)

    assert result == [5, terminal_id]
    assert len(runner.calls) == 2
    output = capsys.readouterr().out
    assert "<5>" in output
    assert f"<{terminal_id}>" not in output
    assert "<7>" not in output


@pytest.mark.parametrize("terminal_id", [3, 4])
def test_generation_falls_back_to_tokenizer_terminal_ids(terminal_id, capsys):
    runner = _Runner([terminal_id, 7])

    result = runner.generate([1, 2], max_seq_len=5, temperature=0)

    assert result == [terminal_id]
    assert len(runner.calls) == 1
    assert f"<{terminal_id}>" not in capsys.readouterr().out


def test_explicit_empty_terminal_ids_run_to_max_length(capsys):
    runner = _Runner([3, 5, 6], eos_ids=[])

    result = runner.generate([1, 2], max_seq_len=5, temperature=0)

    assert result == [3, 5, 6]
    assert len(runner.calls) == 3
    assert "<3><5><6>" in capsys.readouterr().out


def test_explicit_single_terminal_id_overrides_tokenizer(capsys):
    runner = _Runner([3, 5, 7], eos_ids=[5])

    result = runner.generate([1, 2], max_seq_len=6, temperature=0)

    assert result == [3, 5]
    assert len(runner.calls) == 2
    output = capsys.readouterr().out
    assert "<3>" in output
    assert "<5>" not in output


def test_first_generated_terminal_does_not_emit_or_run_model_again(capsys):
    runner = _Runner([200020], eos_ids=[200020, 199999])

    result = runner.generate([1, 2], max_seq_len=5, temperature=0)

    assert result == [200020]
    assert len(runner.calls) == 1
    assert "<200020>" not in capsys.readouterr().out
