# Machine Learning Tutorials

This folder contains Jupyter notebooks, helper scripts, and external resources for learning machine learning with PyTorch.

## Quick Navigation

| Tutorial | Location | Type | Description |
| -------- | -------- | ---- | ----------- |
| **Linear Regression & SGD** | [`../00_linear_regression/`](../00_linear_regression/) | Notebook | Introduction to AI training with linear regression |
| **MNIST with FC Network** | [`../01_mnist/01_mnist_deep_learning.ipynb`](../01_mnist/01_mnist_deep_learning.ipynb) | Notebook + Script | Simple deep learning on handwritten digits |
| **CIFAR-10 with FC Network** | [`../02_cifar10_dnn/02_cifar10_deep_learning.ipynb`](../02_cifar10_dnn/02_cifar10_deep_learning.ipynb) | Notebook + Script | Image classification with fully connected layers |
| **CIFAR-10 with CNN** | [`../03_cifar10_cnn/03_cifar10_cnn.ipynb`](../03_cifar10_cnn/03_cifar10_cnn.ipynb) | Notebook + Script | Convolutional neural networks for image recognition |
| **LLM Basics** | [`../05_llm_basics/04_llm.ipynb`](../05_llm_basics/04_llm.ipynb) | Notebook | Introduction to large language models and transformers |
| **LLM APIs, Embeddings & RAG** | [`./05_llm_apis_and_embeddings.ipynb`](./05_llm_apis_and_embeddings.ipynb) | Notebook | Working with LLM APIs, embeddings, and retrieval-augmented generation |
| **Training a Small LLM** | [`../06_llm_training/06_train_small_llm.ipynb`](../06_llm_training/06_train_small_llm.ipynb) | Notebook | Training a small language model from scratch |
| **Agentic AI Step-by-Step** | [`../08_agentic_ai/TUTORIALS.md`](../08_agentic_ai/TUTORIALS.md) | Scripts + Notes | Agent loops, RAG, MCP-style tools, and skills |

## Advanced Topics

| Topic | Location | Type | Description |
| ----- | -------- | ---- | ----------- |
| **Distributed Training** | [`../04_distributed_training/`](../04_distributed_training/) | Scripts + Notes | Data parallelism with PyTorch DDP (single-node and multi-node) |
| **LLM Distributed Training** | [`../07_llm_distributed_training/`](../07_llm_distributed_training/) | Scripts + Notes | Scaling LLM training with tensor parallelism, pipeline parallelism, and FSDP |
| **Agentic AI** | [`../08_agentic_ai/`](../08_agentic_ai/) | Scripts + Notes | Building AI agents with planning, memory, and tool use |

## Subdirectories

### `./llm-workshop/`

External LLM workshop materials from Argonne National Laboratory (February 2024).

**Contents:**
- `LLMs 101` - Foundational concepts of large language models
- `Prompt Engineering` - Basic and intermediate prompt engineering techniques
- `Retrieval Augmented Generation (RAG)` - Building RAG systems
- `Fine-Tuning` - Fine-tuning existing LLMs
- `LLMs from Scratch` - Training language models from scratch

All notebooks are designed to run on [Google Colaboratory](https://colab.research.google.com/).

See [`./llm-workshop/README.md`](./llm-workshop/README.md) for setup instructions and details.

### `./data/`

Training datasets used by the tutorials:
- **MNIST** - Handwritten digit recognition (28x28 grayscale images, 70K samples)
- **CIFAR-10** - Object recognition (32x32 RGB images, 60K samples)

These datasets are automatically downloaded and cached during notebook execution.

## Helper Scripts

- **`generate_report.py`** - Generates comparison plots and metrics across different models (MNIST FC, CIFAR-10 FC, CIFAR-10 CNN)
- **`build_llm_figures.py`** - Builds visualization figures for LLM concepts

## Running the Tutorials

### On Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload a `.ipynb` file from this directory
3. Set runtime to GPU: `Runtime > Change runtime type > GPU`
4. Run all cells: `Runtime > Run all`

### Locally with Python

```bash
# Install dependencies
pip install torch torchvision jupyter matplotlib numpy

# Start Jupyter
jupyter notebook

# Open and run any .ipynb file
```

For distributed training examples:

```bash
# Single-node MNIST DDP
torchrun --nproc_per_node=2 ../04_distributed_training/train_mnist_ddp.py --epochs 5

# Single-node CIFAR-10 DDP
torchrun --nproc_per_node=2 ../04_distributed_training/train_cifar10_ddp.py --epochs 10

# Generate comparison report
python generate_report.py
```

For the agentic AI tutorials:

```bash
cd ../08_agentic_ai
python3 toy_agent_demo.py --demo explain
python3 mcp_rag_skills_demo.py --demo rag
python3 mcp_stdio_client.py
```

## Learning Path Recommendations

### Beginner (Fundamentals)

1. Start with [`../00_linear_regression/`](../00_linear_regression/) for motivation
2. Work through [`../01_mnist/`](../01_mnist/) to learn basic neural networks
3. Explore [`../03_cifar10_cnn/`](../03_cifar10_cnn/) for convolutional networks

### Intermediate (Large Models)

4. Read [`../05_llm_basics/`](../05_llm_basics/) for transformer basics
5. Try [`./05_llm_apis_and_embeddings.ipynb`](./05_llm_apis_and_embeddings.ipynb) to work with modern LLM APIs
6. Build a small LLM with [`../06_llm_training/`](../06_llm_training/)

### Advanced (Scalability)

7. Study [`../04_distributed_training/`](../04_distributed_training/) for distributed data parallelism
8. Explore [`../07_llm_distributed_training/`](../07_llm_distributed_training/) for LLM-scale distributed training
9. Understand agentic systems with [`../08_agentic_ai/`](../08_agentic_ai/)

## Key Concepts Covered

- **Training vs. Inference**: Understanding the difference and computational costs
- **Loss Functions & Backpropagation**: How neural networks learn
- **Batch Processing & Data Loading**: Efficient training pipelines
- **Convolutional Networks**: Specialized architectures for vision tasks
- **Transformers & Attention**: The foundation of modern LLMs
- **Distributed Training**: Scaling across multiple GPUs/nodes
- **Prompt Engineering & RAG**: Working with pre-trained LLMs
- **Agentic AI**: Building systems that plan, remember, and use tools

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Course Textbook Chapter 19: Machine Learning](../CH19_ML_slides.pdf)

## Software Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision 0.11+
- Jupyter notebook or JupyterLab
- (Optional) CUDA toolkit for GPU acceleration

Standard installation:

```bash
pip install torch torchvision jupyter matplotlib numpy
```

## Notes

- All notebooks are designed to be self-contained and runnable on Google Colab
- GPU acceleration is optional but recommended for faster training
- The LLM workshop materials are external and maintained separately
- Checkpoints and virtual documents are cached Jupyter artifacts (safe to ignore)

---

**Course:** CSCI 394 -- Spring 2026  
**Instructor:** [Huihuo Zheng](https://www.alcf.anl.gov/about/people/huihuo-zheng), Argonne Leadership Computing Facility
