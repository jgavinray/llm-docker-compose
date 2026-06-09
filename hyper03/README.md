# hyper03 Hybrid LLM Deployment

Production-validated hybrid deployment for the RTX PRO 6000 Blackwell host.

## Services

- `llama-cpp`: Gemma 4 31B Q4_K_M GGUF via llama.cpp on host port `8000`
- `vllm`: Qwen3.6 27B AutoRound via patched vLLM on host port `8001`

## Validated Profile

- Qwen/vLLM: `max-model-len=131072`, `max-num-seqs=3`, MTP speculative tokens `5`
- Gemma/llama.cpp: `parallel=3`, `ctx-size=393216`, giving 3 slots at 131072 context each
- Both services resident together: about 86 GB VRAM used, 11 GB free
- Mixed stress test: 5 waves, 30 total generations, 30/30 quality pass
- Aggregate mixed throughput: about 166 completion tokens/sec

## Required Model Files

```text
/storage03/models/Qwen3.6-27B-int4-AutoRound
/storage03/models/gemma4-31b/gguf/gemma-4-31B-it-Q4_K_M.gguf
```

## Run

```bash
docker compose --profile llama up -d --build
```

Health checks:

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8001/health
```

## Notes

The vLLM service is built from the local `Dockerfile` and `entrypoint.sh`.
Those files are part of this deployment, not optional boilerplate.

Compose starts Qwen/vLLM first and waits for its healthcheck before starting
Gemma/llama.cpp. This avoids vLLM profiling/warmup under Gemma's resident memory
pressure.

The legacy NVFP4/DFlash compose has been preserved as:

```text
docker-compose.legacy-nvfp4-dflash.yml
```
