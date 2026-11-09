# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.models.utils import check_logprobs_close
from tests.utils import multi_gpu_test
from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graphs"])
@pytest.mark.parametrize("enable_hc_sp", [False, True], ids=["tp", "hc-sp"])
def test_flash_next_extracts_native_width_features(
    vllm_runner, tmp_path, monkeypatch, enforce_eager, enable_hc_sp
):
    """Feature extraction preserves generation across chunked prefill and TP4."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    target = "Qwen/Qwen3.8-Flash-Next-FP8"
    layers = [4, 24, 44, 48]
    prompts = [
        "Explain how a hash table works.",
        "The following is background information. " * 100 + "Summarize it.",
    ]
    params = SamplingParams(temperature=0, seed=17, max_tokens=8, logprobs=5)
    settings = dict(
        tensor_parallel_size=4,
        enable_hc_sp=enable_hc_sp,
        language_model_only=True,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=2048,
        max_num_batched_tokens=512,
        max_num_seqs=2,
        gpu_memory_utilization=0.8,
    )
    with vllm_runner(target, **settings) as runner:
        reference = runner.generate_w_logprobs(prompts, params)
    with vllm_runner(
        target,
        **settings,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layers},
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(tmp_path)},
        },
    ) as runner:
        outputs = runner.llm.generate(prompts, params)
        actual = []
        for output in outputs:
            sample = output.outputs[0]
            actual.append((list(sample.token_ids), sample.text, sample.logprobs))
            data = example_hidden_states_connector.load_hidden_states(
                output.kv_transfer_params["hidden_states_path"]
            )
            assert data["hidden_states"].shape == (
                len(output.prompt_token_ids),
                len(layers),
                2560,
            )
            assert torch.isfinite(data["hidden_states"]).all()
            assert data["token_ids"].tolist() == output.prompt_token_ids
        assert len(outputs[1].prompt_token_ids) > settings["max_num_batched_tokens"]
    check_logprobs_close(
        outputs_0_lst=actual,
        outputs_1_lst=reference,
        name_0="extract",
        name_1="target",
    )
