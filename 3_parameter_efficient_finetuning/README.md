# Parameter-Efficient Fine-Tuning (PEFT)

This module covers techniques for efficiently adapting large language models using a small number of trainable parameters. PEFT methods allow you to customize models for specific tasks while minimizing computational and memory requirements.

## Introduction

Traditional fine-tuning of large language models presents significant challenges. A full fine-tuning of a 7B parameter model requires substantial GPU memory, makes storing separate model copies expensive, and risks catastrophic forgetting of the model's original capabilities.

PEFT methods address these challenges by training only a small subset of parameters while keeping most of the model frozen. This approach enables fine-tuning on consumer hardware, requires minimal storage for adapter weights, and often leads to better generalization in low-data scenarios.

## Available Methods

### [LoRA (Low-Rank Adaptation)](./lora_adapters.md)
The most widely adopted PEFT method that adds small rank decomposition matrices to model weights. LoRA typically reduces trainable parameters by about 90% while maintaining performance. Learn about:
- Implementation with PEFT library
- Merging adapters for deployment
- Multi-GPU training setup

### [Prompt Tuning](./prompt_tuning.md)
A lightweight approach that adds trainable tokens to inputs rather than modifying model weights. Offers benefits like:
- Minimal memory requirements
- Easy task switching
- Efficient multi-task scenarios

## Getting Started

1. Install required packages:
```bash
pip install transformers peft accelerate
```

2. Choose your PEFT method:
   - For most cases, start with [LoRA](./lora_adapters.md)
   - For very limited resources, try [Prompt Tuning](./prompt_tuning.md)

3. Follow the tutorials in the `notebooks/` directory:
   - `lora_finetuning.ipynb`
   - `prompt_tuning_example.ipynb`

## Resources
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)
- [Hugging Face PEFT Guide](https://huggingface.co/blog/peft)
