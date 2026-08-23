# SFT GPU Memory Probe

All four target lengths completed a real bf16 LoRA-rank-128 forward, backward, optimizer step, and checkpoint save. The base is the same real p99-near trajectory; loss-masked attended suffix tokens stress longer shapes without modifying frozen data.

The 8K/16K DDP probe records exact per-rank torch allocated/reserved peaks. The 24K/32K production fallback uses Ulysses sequence parallel + ZeRO-3 CPU offload; ms-swift exposes allocator peak only for rank 0, so a 200 ms nvidia-smi sampler records actual total device-memory peaks for every GPU. Missing per-rank DeepSpeed allocator fields are labeled unavailable rather than estimated.

## Successful target summary

| Length | Strategy | Forward/backward/step | Rank-0 trainer peak | Step time | Loss | Token accuracy |
|---:|---|---:|---:|---:|---:|---:|
| 8192 | 4-rank DDP + Liger | PASS | 13.49 GiB | 7.4442 s | 0.61973453 | unavailable |
| 16384 | 4-rank DDP + Liger | PASS | 16.89 GiB | 9.5211 s | 0.62159371 | unavailable |
| 24576 | SP4 + ZeRO-3 CPU offload | PASS | 7.65 GiB | 17.7800 s | 0.62500000 | 86.88% |
| 32768 | SP4 + ZeRO-3 CPU offload | PASS | 9.76 GiB | 16.3300 s | 0.60937500 | 86.76% |

## Exact torch allocator records: 8K and 16K

| Length | GPU | Allocated after step MiB | Reserved after step MiB | Peak allocated MiB | Peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 8192 | 0 | 11800.71 | 13902.00 | 13816.71 | 13902.00 |
| 8192 | 1 | 11800.71 | 13902.00 | 13816.71 | 13902.00 |
| 8192 | 2 | 11800.71 | 13902.00 | 13816.71 | 13902.00 |
| 8192 | 3 | 11800.71 | 13902.00 | 13816.71 | 13902.00 |
| 16384 | 0 | 11880.89 | 17942.00 | 17293.64 | 17942.00 |
| 16384 | 1 | 11880.89 | 17782.00 | 17293.64 | 17782.00 |
| 16384 | 2 | 11880.89 | 17782.00 | 17293.64 | 17782.00 |
| 16384 | 3 | 11880.89 | 17782.00 | 17293.64 | 17782.00 |

## Sampled nvidia-smi records: 24K and 32K

These are total device-memory measurements and therefore include the pre-existing ~4.5 GiB allocation on GPU 0. `Peak increment` subtracts the minimum observed baseline.

| Length | GPU | Samples | Min used MiB | Peak used MiB | Peak increment MiB | Min free MiB | Peak util |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24576 | 0 | 343 | 4520 | 13037 | 8517 | 11174 | 100% |
| 24576 | 1 | 343 | 15 | 8532 | 8517 | 15679 | 100% |
| 24576 | 2 | 343 | 15 | 8532 | 8517 | 15679 | 100% |
| 24576 | 3 | 343 | 15 | 8532 | 8517 | 15679 | 100% |
| 32768 | 0 | 311 | 4520 | 15199 | 10679 | 9012 | 100% |
| 32768 | 1 | 311 | 15 | 10694 | 10679 | 13517 | 100% |
| 32768 | 2 | 311 | 15 | 10694 | 10679 | 13517 | 100% |
| 32768 | 3 | 311 | 15 | 10694 | 10679 | 13517 | 100% |

## Decision

The required 32,768-token shape passes on GPUs 0–3 without clearing GPU 0. Frozen training rows have a measured maximum of 7,401 tokens, so formal SFT uses faster four-rank DDP while retaining max_length=32768 and zero truncation. SP4 + ZeRO-3 CPU offload remains the verified fallback for a true 32K row.
