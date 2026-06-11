# vLLM Configuration

The vLLM service is launched by Docker Compose using the `vllm/vllm-openai` image.
It exposes an OpenAI-compatible API used by `VLLMBackendClient`.

## Requirements

- NVIDIA GPU with CUDA support
- `HF_TOKEN` env var set if the model is gated (add to `.env`)

## Changing the model

Edit the `vllm` service command in `infra/docker-compose.yml`:

```yaml
command:
  - --model=Qwen/Qwen3.5-0.8B   # ← change this
  - --max-model-len=4096
  - --gpu-memory-utilization=0.5
```

Then update `configs/models.yaml` to point the `llm` backend at the new model name:

```yaml
backends:
  llm:
    type: vllm
    url: vllm:8000
    model_name: Qwen/Qwen3.5-0.8B   # ← must match --model above
```

## Starting vLLM

```bash
make vllm-up    # starts on host port 8010
make vllm-logs  # tail logs
make vllm-down  # stop
```

## Key env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | _(empty)_ | HuggingFace token for gated models |
| `VLLM_WORKER_MULTIPROC_METHOD` | fork | multiprocessing method |
