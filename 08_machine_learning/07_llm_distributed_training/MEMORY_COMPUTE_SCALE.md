# Memory Footprint, Compute Cost, and Training Time for LLMs

This note answers three concrete questions:

1. **How much memory does LLM training actually require?**
2. **How much compute does it take?**
3. **How long does it take in practice?**

The numbers here are real. Work through them before presenting the parallelism
strategies -- the cost is the motivation.

---

## 1. Why Training Costs More Than Inference

When you run inference on a 7B model, you need approximately:

- 7B × 2 bytes (fp16) ≈ **14 GB** for the weights

That fits comfortably on one A100 (80 GB). Many people assume training costs
roughly the same. It does not.

Training with Adam in mixed precision requires storing:

| Component | Dtype | Bytes per parameter |
| --------- | ----- | ------------------- |
| Parameters (working copy) | fp16 | 2 |
| Gradients | fp16 | 2 |
| Master weights (optimizer) | fp32 | 4 |
| Adam first moment (m) | fp32 | 4 |
| Adam second moment (v) | fp32 | 4 |
| **Total** | | **16** |

So training memory is roughly **8x inference memory** before even counting
activations and temporary buffers.

---

## 2. Memory Breakdown for Real Models

Applying the 16-bytes-per-parameter rule:

| Model | Parameters | Model state only | Activations (estimate) | Total (rough) |
| ----- | ---------- | ---------------- | ---------------------- | ------------- |
| GPT-2 Small | 117M | 1.9 GB | ~0.5 GB | ~2.5 GB |
| GPT-2 XL | 1.5B | 24 GB | ~4 GB | ~28 GB |
| LLaMA-7B | 7B | 112 GB | ~10 GB | ~122 GB |
| LLaMA-13B | 13B | 208 GB | ~18 GB | ~226 GB |
| LLaMA-65B | 65B | ~1.0 TB | ~80 GB | ~1.1 TB |
| GPT-3 | 175B | 2.8 TB | ~200 GB | ~3 TB |

An A100 has 80 GB. A node with 8 × A100 has 640 GB.

- LLaMA-7B needs at least 2 A100s just for model state.
- LLaMA-65B needs at least 14 A100s -- roughly 2 nodes.
- GPT-3 needs roughly 38 A100s -- 5+ nodes.

### Activation memory

Activation memory scales with **batch size × sequence length × model dimension**
and is required during the backward pass. It grows quickly:

- Doubling the batch size doubles activation memory.
- Doubling the sequence length doubles activation memory.
- Longer sequences (e.g., 128K-context models) make activations the
  dominant term.

**Activation checkpointing** (also called gradient checkpointing) trades
activation memory for extra compute by re-running the forward pass in segments
during the backward pass. This is standard in production LLM training.

---

## 3. Why Data Parallelism Alone Is Not Enough

In plain data parallelism (DDP), every GPU holds a **full copy** of the model.

- If the model requires 112 GB and each GPU has 80 GB, DDP cannot even start.
- Adding more GPUs does not help with the memory problem -- each still needs
  the full model.

Data parallelism helps **throughput** (more tokens per second) but not
**capacity** (fitting the model at all).

This is why tensor parallelism, pipeline parallelism, and parameter sharding
(ZeRO / FSDP) were invented.

---

## 4. Compute: How Many Floating-Point Operations?

### FLOPs for one forward pass

For a transformer with:
- `N` parameters
- `L` layers
- `d` model dimension
- `T` sequence length

A good approximation for the forward pass cost is:

```
FLOPs_forward ≈ 2 × N × T
```

The factor of 2 comes from the multiply-accumulate structure of matrix
multiplications (each multiply is paired with an add).

For training (forward + backward):

```
FLOPs_step ≈ 6 × N × T
```

The backward pass costs roughly twice the forward pass (it computes both
activation gradients and weight gradients), so 1 forward + 2 backward ≈ 3×,
and the factor of 2 from the MAC gives 6×.

### Total training FLOPs

To train on `D` tokens:

```
Total FLOPs ≈ 6 × N × D
```

**Chinchilla scaling law** (Hoffmann et al., 2022) shows the compute-optimal
relationship: for a given compute budget `C`, the optimal model size and
dataset size both scale as `√C`. Specifically:

```
N_opt ≈ 0.2 × C^0.5
D_opt ≈ 20 × N
```

Practical implication: most early LLMs (GPT-3, original LLaMA) were
**over-parameterized and under-trained** relative to their compute budget.
The Chinchilla-optimal model for GPT-3's compute budget (3.14×10²³ FLOPs) is
approximately 70B parameters trained on 1.4T tokens -- not 175B on 300B tokens.

---

## 5. Compute for Real Training Runs

### GPT-3 (175B parameters, 300B tokens)

```
Total FLOPs = 6 × 175×10⁹ × 300×10⁹
            = 3.15 × 10²³  FLOPs
```

An A100 at peak fp16 throughput delivers ~312 TFLOPS, but real utilization
in distributed training is typically **30–50%** due to communication, memory
bottlenecks, and pipeline bubbles. Call it ~40 TFLOPS effective.

```
GPU-seconds = 3.15×10²³ / (40×10¹²) ≈ 7.9×10⁹ s
GPU-years   = 7.9×10⁹ / (365×24×3600) ≈ 250 GPU-years
```

OpenAI used ~10,000 V100 GPUs. V100 peak fp16 is 125 TFLOPS (~30 TFLOPS
effective). At that scale the run took roughly 2–3 weeks.

### LLaMA-1 (65B parameters, 1.4T tokens)

Meta reported using **82,432 A100 GPU-hours** for the 65B model, or roughly
**9.4 GPU-years**. The Chinchilla-optimal training budget was larger than
GPT-3 per parameter because of the longer token count.

### LLaMA-2 (70B parameters, 2T tokens)

Meta reported **1,720,320 GPU-hours** of A100 compute across all model sizes,
with the 70B model taking approximately **2 million GPU-hours** on its own --
roughly **228 GPU-years**.

---

## 6. GPU Utilization and the Roofline

Not all compute is equal. The key metric for LLM training efficiency is
**Model FLOPs Utilization (MFU)**:

```
MFU = (actual FLOPs/s) / (theoretical peak FLOPs/s)
```

Typical MFU for well-tuned LLM training:

| Setup | MFU |
| ----- | --- |
| Naive PyTorch, single GPU | 10--25% |
| Flash Attention + AMP | 35--45% |
| Production (Megatron-LM) | 45--60% |

Low MFU means the GPU is waiting -- either for memory bandwidth (memory-bound)
or for communication (communication-bound). Improving MFU is one of the main
goals of LLM systems research.

---

## 7. Time and Cost at a Glance

| Model | Tokens | GPU-hours (A100) | Calendar time (1k A100s) | Approx cost |
| ----- | ------ | ---------------- | ------------------------ | ----------- |
| GPT-2 (1.5B) | 40B | ~500 | ~0.5 hours | ~$1,000 |
| LLaMA-7B | 1T | ~35,000 | ~35 hours | ~$100K |
| LLaMA-65B | 1.4T | ~82,000 | ~82 hours | ~$250K |
| GPT-3 (175B) | 300B | ~3.6M | ~150 days | ~$5--10M |
| LLaMA-2 70B | 2T | ~2M | ~83 days | ~$5M |

*Cost at ~$3 per A100-hour (cloud spot pricing, 2024).*

These numbers explain why foundation model training is concentrated at large
organizations -- and why efficient distributed training systems are among the
most valuable software in modern AI.

---

## 8. Takeaways

- Training memory is **8x inference memory** for Adam in mixed precision.
- Data parallelism alone cannot solve the memory problem for large models.
- Total training FLOPs scale as `6 × N × D`.
- Chinchilla scaling: the compute-optimal training ratio is roughly 20 tokens
  per parameter.
- Real MFU in production is 40--60%. The gap is communication and memory
  bandwidth.
- GPT-3-scale training requires millions of GPU-hours and tens of millions of
  dollars.

---

## Exercises

1. How much memory does a 13B-parameter model require during training with
   AdamW in fp16 mixed precision? Does it fit on 4 × A100-80GB?

2. A team has 256 A100 GPUs for 30 days. How many training tokens can they
   afford for a 7B-parameter model, assuming 40% MFU?

3. Using the Chinchilla rule, what is the compute-optimal model size for a
   budget of 10²² FLOPs?

4. Why does doubling the sequence length increase activation memory but not
   model state memory?

5. If you add gradient checkpointing and reduce activation memory by 75%, how
   does this change the minimum number of GPUs needed to train LLaMA-65B?
