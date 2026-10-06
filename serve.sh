#!/usr/bin/env bash
# serve.sh — OpenAI-compatible endpoint for Qwen3.8-27B (qwen35, hybrid SSM + attention)
#             on a 12 GB GPU.
#
# Measured on: Ryzen 9 9900X · RTX 4070 SUPER 12 GB · 30 GB RAM
#   16.8 tok/s decode median over 163 requests (18.1-18.4 on a 30-token prompt, 17.0 at 4-11k,
#   13.5 at 57k) · 605 tok/s prefill on a 4-11k prompt, 533 on a 57k prompt · 64k context ·
#   11,724 MiB idle, 11,947 peak of 12,282 (7 free at peak) · RSS 2.1-2.2 GiB idle → 4.0 GiB
#
# Override with env vars:  MODEL=… LLAMA_SERVER=… PORT=… ./serve.sh
# Extra llama-server args pass straight through:  ./serve.sh -ngl 56
set -euo pipefail

MODEL="${MODEL:-$HOME/models/qwen3.8-27b/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/src/llama.cpp/build-cuda/bin/llama-server}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8105}"

[ -f "$MODEL" ]   || { echo "serve.sh: model not found: $MODEL" >&2; exit 1; }
[ -x "$LLAMA_SERVER" ] || { echo "serve.sh: llama-server not found: $LLAMA_SERVER (run ./build.sh)" >&2; exit 1; }

# -ngl 58            58 of 64 blocks resident = 9,072 MiB. 64 OOMs (~500 MiB over the card).
# -ot token_embd=CPU moves the 388 MiB embedding table out of VRAM; effectively free.
# -ctk/-ctv q4_0     KV = 1,125 MiB @64k instead of 4,000 for f16. Only type that fits.
# -c 64000           sized to that KV budget.
# --parallel 1       one slot: KV allocated once, not per slot.
# -b 2048 -ub 96     physical batch 96 keeps the compute graph inside the ~845 MiB
#                    of residual VRAM. Raise for prefill speed, re-measure headroom.
# -t 12              host threads for the CPU-resident layers (6 for the stock file,
#                    7 if you serve the -mtp build, whose blk.64 stays on the host).
exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --parallel 1 -c 64000 -b 2048 -ub 96 -t 12 \
  -ctk q4_0 -ctv q4_0 \
  -ot token_embd=CPU \
  -ngl 58 \
  "$@"
