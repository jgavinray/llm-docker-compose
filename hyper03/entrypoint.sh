#!/bin/bash
# vLLM startup for Qwen3.6 on RTX PRO 6000 Blackwell (96GB)
# Single-GPU max-throughput profile for Qwen3.6-27B INT4 AutoRound.
set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_FLOAT32_MATMUL_PRECISION=high
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=8
unset PYTORCH_CUDA_ALLOC_CONF

# ── Configurable defaults ────────────────────────────────────────────────────
MODEL="${MODEL:-/models/Qwen3.6-27B-int4-AutoRound}"
MODEL_NAME="${MODEL_NAME:-qwen3.6-27b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.523}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-3}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"
PORT="${PORT:-8000}"

echo "Starting vLLM: model=$MODEL, name=$MODEL_NAME, context=$MAX_MODEL_LEN, mem=$GPU_MEM_UTIL, MTP n=$NUM_SPEC_TOKENS"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --quantization auto_round \
  --attention-backend flash_attn \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --language-model-only \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}"
