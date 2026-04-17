# Large Language Models (LLMs)

**CSCI 394 -- Spring 2026**

---

## 1. What is a Language Model?

A **language model** is a system that predicts the probability of the next
token (word or sub-word) given the preceding tokens.  Given the sequence
"The cat sat on the", a language model assigns probabilities to possible
next tokens:

| Next token | Probability |
| ---------- | ----------- |
| mat        | 0.25        |
| floor      | 0.18        |
| roof       | 0.07        |
| ...        | ...         |

A **large** language model (LLM) is simply a language model with billions
of parameters, trained on trillions of tokens of text from books, websites,
code, and other sources.

![Tokenization pipeline](figures/llm_tokenization.svg)

---

## 2. From RNNs to Transformers: A Brief History

| Year | Model / Paper | Key idea |
| ---- | ------------- | -------- |
| 2013 | Word2Vec | Dense word embeddings via skip-gram / CBOW |
| 2014 | Seq2Seq + Attention (Bahdanau) | Encoder-decoder with attention for machine translation |
| 2017 | **Transformer** ("Attention Is All You Need") | Self-attention replaces recurrence entirely |
| 2018 | GPT-1 (117M params) | Decoder-only transformer, pre-train then fine-tune |
| 2018 | BERT (340M params) | Encoder-only, masked language modeling |
| 2019 | GPT-2 (1.5B params) | Scaling up decoder-only pre-training |
| 2020 | GPT-3 (175B params) | In-context learning, few-shot prompting |
| 2022 | ChatGPT / InstructGPT | RLHF -- aligning models to follow instructions |
| 2023 | Llama 2, Mixtral, Gemini | Open-weight models, mixture-of-experts |
| 2024 | Llama 3, Claude 3, GPT-4o | Multimodal, longer context, better reasoning |
| 2025 | DeepSeek-R1, Claude 4, Llama 4 | Reasoning models, extended thinking |

The single most important architectural breakthrough was the **Transformer**
(Vaswani et al., 2017), which made it feasible to train on very long sequences
in parallel on GPUs.

---

## 3. The Transformer Architecture

### 3.1 High-Level View

A Transformer consists of stacked layers, each containing:

1. **Multi-Head Self-Attention** -- lets each token attend to all other tokens
2. **Feed-Forward Network (FFN)** -- a position-wise MLP applied independently
   to each token
3. **Layer Normalization + Residual Connections** -- stabilize training

![Transformer block diagram](figures/llm_transformer_block.svg)

### 3.2 Self-Attention

Self-attention computes, for each token, a weighted combination of all tokens
in the sequence.  The weights are determined by how "relevant" other tokens
are to the current one.

Given an input matrix **X** (sequence length x model dimension):

1. Project into **Q** (query), **K** (key), **V** (value):

   Q = X W_Q,  K = X W_K,  V = X W_V

2. Compute attention scores:

   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

The `sqrt(d_k)` scaling prevents dot products from growing too large.

![Self-attention mechanism](figures/llm_self_attention.svg)

**Multi-head attention** runs `h` independent attention operations in parallel,
each with its own Q/K/V projections, then concatenates the results.  This lets
the model attend to different kinds of relationships simultaneously (e.g.,
syntactic, semantic, positional).

### 3.3 Positional Encoding

Unlike RNNs, Transformers process all tokens in parallel and have no built-in
notion of order.  **Positional encodings** are added to the input embeddings to
give the model information about token positions.  Options include:

- **Sinusoidal** (original Transformer): fixed sine/cosine functions at
  different frequencies
- **Learned** (GPT-2): a trainable embedding table, one vector per position
- **Rotary (RoPE)** (Llama, modern models): rotation-based encoding that
  naturally captures relative position

### 3.4 Decoder-Only vs Encoder-Only vs Encoder-Decoder

| Variant | Masking | Use case | Examples |
| ------- | ------- | -------- | -------- |
| **Decoder-only** | Causal (can only attend to past tokens) | Text generation, chatbots | GPT, Llama, Claude |
| **Encoder-only** | Bidirectional (attends to all tokens) | Classification, embeddings | BERT, RoBERTa |
| **Encoder-decoder** | Encoder is bidirectional, decoder is causal | Translation, summarization | T5, BART |

Most modern LLMs (GPT-4, Claude, Llama) are **decoder-only**.

![Causal vs bidirectional masking](figures/llm_causal_mask.svg)

---

## 4. Training LLMs

### 4.1 Pre-training

The model is trained on a massive text corpus with a simple objective:
**predict the next token**.

```
Input:  "The cat sat on the"
Target: "cat sat on the mat"
```

This is called **causal language modeling** (CLM).  The loss is cross-entropy
over the vocabulary at each position.

**Scale of pre-training:**

| Model | Parameters | Training tokens | Training compute (FLOPs) |
| ----- | ---------- | --------------- | ------------------------ |
| GPT-3 | 175B | 300B | ~3.6 x 10^23 |
| Llama 2 70B | 70B | 2T | ~1.7 x 10^24 |
| Llama 3 405B | 405B | 15T | ~4 x 10^25 |

The compute required scales roughly as `C ≈ 6 * N * D` where N is the number
of parameters and D is the number of training tokens (the Chinchilla scaling
law).

### 4.2 Fine-tuning

After pre-training, the model is adapted for specific tasks:

- **Supervised Fine-Tuning (SFT)**: train on curated (instruction, response)
  pairs
- **RLHF (Reinforcement Learning from Human Feedback)**: train a reward model
  from human preference comparisons, then optimize the LLM policy via PPO or
  DPO to maximize the reward

![LLM training pipeline](figures/llm_training_stages.svg)

### 4.3 Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning of a 70B-parameter model requires hundreds of GBs of GPU
memory.  PEFT methods reduce this dramatically:

- **LoRA (Low-Rank Adaptation)**: freeze the original weights; add small
  trainable low-rank matrices to each layer.  Typically < 1% of the total
  parameters are trainable.
- **QLoRA**: combine LoRA with 4-bit quantization of the base model

![LoRA diagram](figures/llm_lora.svg)

---

## 5. Inference and Serving

### 5.1 Autoregressive Generation

LLMs generate text one token at a time:

1. Feed the prompt through the model (the "prefill" phase)
2. Sample or greedily select the next token
3. Append it to the context and repeat

This makes generation inherently **sequential** -- each new token depends on
all previous tokens.

![Autoregressive generation with KV cache](figures/llm_autoregressive.svg)

### 5.2 Decoding Strategies

| Strategy | Description |
| -------- | ----------- |
| Greedy | Always pick the highest-probability token |
| Top-k | Sample from the top k most probable tokens |
| Top-p (nucleus) | Sample from the smallest set of tokens whose cumulative probability >= p |
| Temperature | Scale logits by 1/T before softmax (T<1 = sharper, T>1 = more random) |

### 5.3 KV Cache

During generation, the key and value tensors from previous tokens are cached
so that each new token only needs to compute attention with one new query
vector instead of reprocessing the entire sequence.

**Memory cost**: KV cache size = `2 * n_layers * n_heads * d_head * seq_len * batch_size * bytes_per_param`

For a 70B model with 80 layers and 8192 context, the KV cache alone can
consume tens of GBs of memory.

### 5.4 Quantization

Reduce model memory by storing weights in lower precision:

| Precision | Bytes/param | 70B model size |
| --------- | ----------- | -------------- |
| FP32      | 4           | 280 GB         |
| FP16/BF16 | 2           | 140 GB         |
| INT8      | 1           | 70 GB          |
| INT4      | 0.5         | 35 GB          |

Tools like GPTQ, AWQ, and bitsandbytes make quantization straightforward.

---

## 6. Emergent Capabilities

As LLMs scale, they exhibit capabilities that smaller models do not:

- **In-context learning**: perform new tasks from a few examples in the
  prompt, without any weight updates
- **Chain-of-thought reasoning**: solving multi-step problems by generating
  intermediate reasoning steps
- **Code generation**: writing, debugging, and explaining code
- **Tool use**: calling external APIs, calculators, or databases

These capabilities generally appear as the model crosses certain parameter
and training-data thresholds, though the exact scaling behavior is an active
area of research.

---

## 7. Running LLMs on HPC Systems

### 7.1 Why Supercomputers?

| Requirement | Typical scale |
| ----------- | ------------- |
| Model memory | 35--560 GB (depending on precision) |
| Training compute | 10^24 -- 10^25 FLOPs |
| Training data | Terabytes of text |
| Training time | Weeks to months on thousands of GPUs |

A single GPU (e.g., NVIDIA A100 with 80 GB) cannot hold a 70B model.
Training and even inference require **model parallelism** across multiple
GPUs or accelerators.

### 7.2 Parallelism Strategies for LLM Training

| Strategy | What is split | When to use |
| -------- | ------------- | ----------- |
| **Data parallelism** | Each GPU holds a full model copy; data is split | Model fits in one GPU |
| **Tensor parallelism** | Individual layers are split across GPUs | Within a node (high bandwidth) |
| **Pipeline parallelism** | Different layers on different GPUs | Across nodes |
| **Fully Sharded Data Parallelism (FSDP)** | Weights, gradients, and optimizer states are sharded | Large models; PyTorch native |

Modern LLM training typically combines all three: FSDP or ZeRO across nodes,
tensor parallelism within a node, and pipeline parallelism for the deepest
models.

![Parallelism strategies](figures/llm_parallelism.svg)

### 7.3 LLM Inference on Aurora

Aurora has Intel Data Center GPU Max (Ponte Vecchio) accelerators.  You can
run open-weight LLMs using:

```bash
# Using the frameworks module
module load frameworks

# Example: run a Hugging Face model with PyTorch
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'meta-llama/Llama-3.2-1B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
).to('xpu')

inputs = tokenizer('The future of supercomputing is', return_tensors='pt').to('xpu')
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
"
```

For larger models (7B+), you will need multiple tiles or nodes, and frameworks
like DeepSpeed or vLLM with Intel XPU support.

---

## 8. Prompt Engineering

Prompt engineering is the practice of designing inputs to get the best outputs
from an LLM.

### Key Techniques

| Technique | Description | Example |
| --------- | ----------- | ------- |
| **Zero-shot** | Ask directly, no examples | "Translate to French: Hello" |
| **Few-shot** | Provide examples in the prompt | "English: Hello -> French: Bonjour\nEnglish: Goodbye -> French:" |
| **Chain-of-thought** | Ask the model to reason step by step | "Solve step by step: If a train..." |
| **System prompts** | Set the model's role and constraints | "You are a helpful physics tutor..." |

### Tips

- Be specific and explicit about what you want
- Provide context and constraints
- For complex tasks, break them into subtasks
- Iterate: prompt engineering is empirical

---

## 9. Limitations and Risks

- **Hallucination**: LLMs can generate confident but factually incorrect text.
  They are pattern matchers, not knowledge databases.
- **Context window limits**: models can only process a finite number of tokens
  at a time (e.g., 8K, 128K, or 1M tokens depending on the model).
- **Bias**: training data biases are reflected in model outputs.
- **Cost**: training a frontier LLM costs tens of millions of dollars in
  compute alone.
- **Environmental impact**: training runs consume enormous amounts of energy.

---

## 10. Beyond Generation: Retrieval-Augmented Generation (RAG)

RAG combines a vector search index with an LLM to ground answers in real documents,
reducing hallucinations and allowing the model's knowledge to be updated without
retraining.

![RAG pipeline](figures/llm_rag.svg)

See `05_llm_apis_and_embeddings.ipynb` for a hands-on RAG implementation.

---

## 11. Hands-On Exercises

### Exercise 1: Tokenization

Explore how text is converted to tokens using different tokenizers:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
text = "Supercomputing enables breakthroughs in science."
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)
print(f"Text:   {text}")
print(f"Tokens: {tokens}")
print(f"IDs:    {ids}")
print(f"Vocab size: {tokenizer.vocab_size}")
```

### Exercise 2: Measuring Generation Throughput

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
).to("xpu")

prompt = "Explain the concept of parallel computing in three sentences."
inputs = tokenizer(prompt, return_tensors="pt").to("xpu")

torch.xpu.synchronize()
t0 = time.perf_counter()
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
torch.xpu.synchronize()
t1 = time.perf_counter()

generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
print(f"Generated {generated_tokens} tokens in {t1-t0:.2f}s")
print(f"Throughput: {generated_tokens/(t1-t0):.1f} tokens/s")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Exercise 3: Prompt Engineering

Try different prompting strategies on the same task and compare outputs:

1. Zero-shot: "What is MPI?"
2. Few-shot: provide 2 examples of similar definitions, then ask
3. Chain-of-thought: "Explain step by step how MPI_Send works"
4. Role-based: "You are a parallel computing expert teaching undergraduates..."

---

## 11. Further Reading

- Vaswani et al., "Attention Is All You Need" (2017) -- the original
  Transformer paper
- Brown et al., "Language Models are Few-Shot Learners" (2020) -- GPT-3
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023)
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models"
  (Chinchilla, 2022)

---

## Key Takeaways

1. LLMs are Transformer-based models trained to predict the next token on
   massive text corpora.
2. The Transformer's self-attention mechanism is the core innovation that
   enables parallelism and long-range dependencies.
3. Training happens in stages: pre-training (next-token prediction),
   supervised fine-tuning, and alignment (RLHF).
4. Inference is autoregressive and memory-bound; KV caching and quantization
   are essential optimizations.
5. Running LLMs at scale requires the same HPC parallelism techniques
   (data, tensor, pipeline) that we studied for distributed training.
6. LLMs are powerful but imperfect -- always verify their outputs.
