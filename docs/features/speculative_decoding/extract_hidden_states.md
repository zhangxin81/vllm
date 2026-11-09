# Hidden State Extraction

The Hidden State Extraction feature allows vLLM to save intermediate layer activations from a target model during inference. This is useful for training [EAGLE](eagle.md)-style draft models, knowledge distillation, or offline analysis of model internals.

!!! note
    It is possible to save the last-layer's output hidden states by passing `num_hidden_layers` as a layer id. Note that these are _not_ normalized using the output norm.

## Offline Example

### Qwen3.8-Flash-Next feature layout

On NVIDIA, Flash Next can expose native-width features with Model Runner V2
and `pipeline_parallel_size=1`. Engine layer index `L` captures the 2560-wide
input to layer `L`'s attention, after its learned HyperConnection read and PLE
addition. The final index `48` captures the final mixer's language-model-head
input. These are not the 10240-wide multi-stream states used by native MTP.

For example, set `eagle_aux_hidden_state_layer_ids` to `[4, 24, 44, 48]` to
capture three intermediate features and the final head input. A draft trained
on these features must use the same capture points during inference. The final
feature already includes the final mixer's normalization; do not apply a
second output norm when forming training target logits.

The saved tensor has shape `[num_tokens, num_selected_layers, 2560]`, including
with TP, HC sequence parallelism and chunked prefill. Intermediate capture
reuses the attention input already produced by the layer's HC read and SP
gather; it adds no HC projection, gather or feature clone. AMD and auxiliary
feature transport over PP are not covered by this implementation.

```python
import tempfile

from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)

with tempfile.TemporaryDirectory() as tmpdir:
    llm = LLM(
        model="Qwen/Qwen3-8B",
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4],
                },
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": tmpdir,
            },
        ),
    )

    outputs = llm.generate(
        ["The future of AI is"],
        SamplingParams(max_tokens=1),
    )

    for output in outputs:
        path = output.kv_transfer_params["hidden_states_path"]
        obj = example_hidden_states_connector.load_hidden_states(path)
        print(f"token_ids: {obj['token_ids'].shape}")
        print(f"hidden_states: {obj['hidden_states'].shape}")
```

A complete example is available at [`examples/features/speculative_decoding/extract_hidden_states_offline.py`](../../../examples/features/speculative_decoding/extract_hidden_states_offline.py).

## Online Example

For improved performance, it is recommended to use a RAM-mounted file system such as `/dev/shm/` for online usage in which the client cleans up the files soon after they are generated.

```bash
vllm serve Qwen/Qwen3-8B \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4]}}}' \
    --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/dev/shm/hidden_states"}}'
```

## Per-Request Options

Both offline and online modes support per-request options via `kv_transfer_params`:

| Parameter | Default | Description |
| --- | --- | --- |
| `hidden_states_path` | Auto-generated | Custom file path for saving hidden states. If not set, files are saved to `<shared_storage_path>/<request_id>.safetensors`. Requires `allow_custom_save_path` to be enabled in the server config. |
| `include_output_tokens` | `False` | When `True`, save hidden states for both prompt and generated output tokens. When `False`, only prompt token hidden states are saved. |

### Offline usage

Pass per-request options via `extra_args` on `SamplingParams`:

```python
SamplingParams(
    max_tokens=32,
    extra_args={
        "kv_transfer_params": {
            "hidden_states_path": "/tmp/my_output.safetensors",
            "include_output_tokens": True,
        }
    },
)
```

### Online usage

Pass `kv_transfer_params` as a top-level field in the API request:

```json
{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32,
    "kv_transfer_params": {
        "hidden_states_path": "/tmp/my_output.safetensors",
        "include_output_tokens": true
    }
}
```

## Configuration

The `kv_connector_extra_config` dict accepts these server-level options:

| Parameter | Default | Description |
| --- | --- | --- |
| `shared_storage_path` | `/tmp` | Directory where hidden state files are saved (used when `hidden_states_path` is not set per-request) |
| `allow_custom_save_path` | `False` | Allow API clients to specify custom file paths via `hidden_states_path`. When disabled, client-provided paths are ignored with a warning. Enable only with trusted clients — custom paths can write to arbitrary locations on the server. |
| `num_writer_threads` | `8` | Thread pool size for async disk writes |
| `use_synchronization_lock` | `True` | Use file locks so concurrent readers block until writes complete. Can be disabled for batch generation where synchronization is not needed. |

## Output Format

Each request produces a `.safetensors` file containing:

- **`hidden_states`** — shape `[num_tokens, num_extracted_layers, hidden_size]`
- **`token_ids`** — shape `[num_tokens]`

The file path is returned in `output.kv_transfer_params["hidden_states_path"]`. Use `load_hidden_states()` from the connector module to read the file with proper synchronization.

!!! note
    Chunked prefill is not compatible with this feature and must be disabled.
