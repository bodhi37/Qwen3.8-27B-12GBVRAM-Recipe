Qwen3.8-27B-GSQ-RCO-Heretic (IQ3_S) @ 64k context
~16-18 tok/sec decode, ~280-600 tok/sec prefill
on just 12GB VRAM + 8GB RAM

| metric | measured |
|---|---|
| Decode | 16.8 tok/s median over 163 requests (p90 18.0, max 18.5). |
| Prefill | 280 tok/s median over the same 163 (max 673). 605 on a 4-11k prompt, 533 on a 57k prompt |

---

## Models

Same flags, same budget for both. Pick one:

| build | repo | file | size |
|---|---|---|---:|
| Stock | [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` | 11.0 GiB |
| Abliterated | [0bserverx/Qwen3.8-27B-Heretic-GSQ-RCO-GGUF](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-GSQ-RCO-GGUF) | `RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_S.gguf` | 11.0 GiB |
| Abliterated + MTP head (what this box runs) | same repo | `RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_S-mtp.gguf` | 11.40 GiB |

The abliterated file is the ISTA quant with weights edited in place. All 851 tensors match stock except 48 `ssm_alpha` tensors at F32 instead of BF16 (+22.5 MiB), plus one extra block `blk.64` (draft head, 15 tensors, 430 MiB). At `-ngl 58` that leaves 7 blocks on the host instead of 6.

| metric | Stock `IQ3_S` | Abliterated `-mtp` |
|---|---:|---:|
| Prefill | 600 tok/s | 564 tok/s (-6%) |
| Decode | 16.8 tok/s | 15.6 tok/s (-7%) |
| VRAM idle to peak | 11724 to 11932 MiB | 11758 to 11758 MiB |
| Free at peak | 22 MiB | 196 MiB |
| RSS peak | 3.9 GiB | 4.3 GiB |

---

## Hardware

Reference box: Ryzen 9 9900X 12C/24T, RTX 4070 SUPER 12 GB (12282 MiB), 30.4 GiB RAM (8 GB is enough), NVMe, Linux, GCC 16.2.1, CUDA 13.4 toolkit.

---

## Why it fits

### 1. Only 16 of 64 layers need KV cache

Qwen3.8-27B is hybrid SSM + attention (`general.architecture = qwen35`):

| layer type | count | KV cache |
|---|---:|---|
| SSM / linear attention | 48 | no, recurrent state only |
| Full attention (3, 7, 11, ... 63) | 16 | yes, every 4th layer |

An all-attention model with these dims would need about 16 GB of `f16` KV at 64k. This one needs 1.1 GB:

```
16 layers x 4 KV heads x 256 head_dim x 2 (K+V) = 32768 elements/token
q4_0 = 18 bytes / 32 elements                   = 18432 bytes/token
x 64000 tokens                                  = 1125 MiB
```

| KV type | at 64k | fits |
|---|---:|---|
| `f16` | 4000 MiB | no |
| `q8_0` | 2125 MiB | no |
| `q4_0` | 1125 MiB | yes |

`-ctk q4_0 -ctv q4_0` is what makes 64k load.

### 2. 11 GiB of weights, split over PCIe

`IQ3_S` averages 3.5 bits/weight. File is 11771546784 bytes (11.0 GiB). Sizes from the GGUF header:

| tensor group | size | where |
|---|---:|---|
| Blocks 0-57 | 9072 MiB | GPU (`-ngl 58`) |
| `output.weight` + `output_norm` | 682 MiB | GPU |
| Blocks 58-63 | 1073 MiB | CPU |
| `token_embd.weight` | 388 MiB | CPU (`-ot token_embd=CPU`) |

Embedding lookup runs once per request and stays mmapped, so offloading it saves 388 MiB for no measurable cost.

### 3. Total budget

| component | MiB |
|---|---:|
| Weights on GPU (blocks 0-57 + output) | 9754 |
| KV cache (`q4_0`, full 64k) | 1125 |
| CUDA context + graph (idle residual) | ~845 |
| Measured idle | 11724 |
| Measured peak (first-use workspace + prompt cache) | 11947 |
| Free at peak | 7 |

Two notes on that last row. `12282 - used` overstates free; the driver holds about 330 MiB outside it. Peak is what matters. Observed peaks were 11916-11947 MiB (7-38 MiB free) depending on request. It never OOMed here, but there is no room left for larger `-ub`, a second slot, or longer context.

- `-ngl 64` OOMs. Blocks 58-63 add 1073 MiB: 10827 + 1125 KV = 11952 MiB before context, graphs, and driver overhead, about 500 MiB over. It fails on KV alloc: `allocating 1125.00 MiB on device 0: cudaMalloc failed: out of memory` then `failed to allocate buffer for kv cache`.
- `-ot token_embd=CPU` is required to start at all.

### 4. RAM stays under 8 GB

CPU blocks plus embedding table are 1461 MiB of weights, mmapped. RSS: 2.1-2.2 GiB idle, 2.8 GiB after one 57k request, 4.0 GiB after three 10801-token requests (3.9-4.3 GiB across builds). Grows about 0.6 GiB per cached prompt. That is prompt cache, not weights. Budget 8 GB for a server holding several long contexts. At idle `smaps_rollup` shows 1598 MiB file-backed plus 525 MiB anonymous.

---

## Quality

BF16 is 53.8 GB. On 12 GB VRAM you quantize regardless. Question is which quant.

GSQ-RCO ([GSQ](https://arxiv.org/abs/2604.18556), [RCO](https://arxiv.org/abs/2605.00649), ISTA DASLab) is non-uniform: each tensor gets its own quant type from a gradient search under a size budget. Sensitive tensors keep precision, redundant ones lose it.

Published numbers, `IQ3_S` vs BF16 (sizes decimal GB; the 11.8 GB row is the 11.0 GiB file used here):

| variant | bpw | size | Wiki ppl | ZS avg | Recovery | AIME25 | GPQA-D | LCB v6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 16.00 | 53.8 GB | 7.05 | 74.34 | 100.0% | 100.00 | 89.90 | 85.71 |
| GSQ-RCO IQ3_S (this recipe) | 3.50 | 11.8 GB | 7.07 | 74.47 | 100.2% | 100.00 | 89.39 | 85.71 |
| UD-IQ3_S (uniform ref) | 3.52 | 12.0 GB | 7.16 | 75.49 | 101.5% | 96.67 | 89.90 | 84.00 |

Task average 91.70 vs 91.87. Ties AIME25 and LiveCodeBench, -0.51 on GPQA-Diamond.

---

## Reproduce

### 1. Model

```bash
# Stock, IQ3_S (11.0 GiB)
hf download ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF \
  Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf --local-dir ~/models/qwen3.8-27b

# Abliterated, same quant (11.0 GiB; -mtp twin 11.40 GiB)
hf download 0bserverx/Qwen3.8-27B-Heretic-GSQ-RCO-GGUF \
  RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_S.gguf --local-dir ~/models/qwen3.8-27b
```

Name the file exactly. `--include "*IQ3_S*"` also pulls the `-mtp` twin. Other sizes in the ISTA repo: `IQ3_XXS` (10.1 GB, smaller/faster), `IQ2_XS` / `IQ2_S` (8.4 and 9.3 GB, clear quality cost). `mmproj` adds vision; this recipe is text-only.

Both builds use the same flags below. Any other `qwen35` Qwen3.8-27B GGUF up to about 11 GiB is a drop-in.

### 2. Build

```bash
./build.sh        # pins commit 254b177, flags in the script
```

That commit is clean upstream. Pin matters: buffer sizes and `--fit` change between releases, so retune `-ngl` if you move.

### 3. Serve

```bash
./serve.sh                                       # defaults, or:
MODEL=/path/to/model.gguf PORT=8105 ./serve.sh
```

Verbatim:

```bash
llama-server \
  -m  ~/models/qwen3.8-27b/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf \
  --host 127.0.0.1 --port 8105 \
  --parallel 1 -c 64000 -b 2048 -ub 96 -t 12 \
  -ctk q4_0 -ctv q4_0 \
  -ot token_embd=CPU \
  -ngl 58
```

Abliterated differs only in `-m`:

```bash
  -m  ~/models/qwen3.8-27b/RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_S-mtp.gguf
```

| flag | reason |
|---|---|
| `-ngl 58` | 58 of 64 blocks on GPU = 9072 MiB. 64 OOMs. Drop by one on smaller cards |
| `-ot token_embd=CPU` | 388 MiB embedding table off GPU |
| `-ctk q4_0 -ctv q4_0` | KV 1125 MiB instead of 4000. Needed for 64k. Third-party A/Bs on other models show about +0.4% perplexity; not remeasured here |
| `-c 64000` | Sized to that KV budget |
| `--parallel 1` | One slot. KV allocated once; extra clients queue instead of using more memory |
| `-b 2048 -ub 96` | Physical batch 96 keeps the graph inside the ~845 MiB residual. Higher is faster prefill, less headroom |
| `-t 12` | Threads for host layers. Half the physical cores; SMT siblings measured worse |
| `--host 127.0.0.1` | Loopback only. Add `--api-key` and set CORS if you expose it |

Flash attention stays at default `auto`, which is on with quantized V cache. `-fa off -ctv q4_0` fails with `quantized V cache requires flash_attn to be enabled`. No `-fa` flag is needed. CUDA graphs are compiled in (`GGML_CUDA_GRAPHS=ON`, upstream default `OFF`). The server logs `graphs reused` per request when active.

## Measuring it yourself

- Decode/prefill: `print_timing` lines in the server log, e.g. `eval time = 47637.25 ms / 809 tokens (58.96 ms per token, 16.96 tokens per second)`. Decode depends on context: 18.1-18.4 tok/s on a 30-token prompt, 17.0 on 4-11k, 16.8 across 163 requests, 13.5 at 57k.
- VRAM: sample peak during a request, not idle: `nvidia-smi --query-gpu=memory.used,memory.free --format=csv` once per second.
- Tensor sizes: read the GGUF header. 851 tensors with type and dims. Bytes = `elements x bits/8 / block_size`; consecutive data offsets give the same answer. Keys: `qwen35.block_count`, `qwen35.attention.head_count_kv`, `qwen35.attention.key_length`, `qwen35.full_attention_interval`.

---

## License

MIT for these scripts, see [LICENSE](LICENSE). Models are Apache-2.0 per Hugging Face metadata. Engine is [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) under its own license; this recipe only pins it.
