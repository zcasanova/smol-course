# Parameter-Efficient Fine-Tuning (PEFT)

As language models grow larger, traditional fine-tuning becomes increasingly challenging. A full fine-tuning of a 7B parameter model requires substantial GPU memory, makes storing separate model copies expensive, and risks catastrophic forgetting of the model's original capabilities. Parameter-efficient fine-tuning (PEFT) methods address these challenges by modifying only a small subset of model parameters while keeping most of the model frozen.

## Understanding PEFT

Traditional fine-tuning updates all model parameters during training, which becomes impractical for large models. PEFT methods introduce innovative approaches to adapt models using far fewer trainable parameters - often less than 1% of the original model size. This dramatic reduction in trainable parameters enables:

- Fine-tuning on consumer hardware with limited GPU memory
- Storing multiple task-specific adaptations efficiently
- Better generalization in low-data scenarios
- Faster training and iteration cycles

## Available Methods

### [LoRA (Low-Rank Adaptation)](./lora_adapters.md)

LoRA has emerged as the most widely adopted PEFT method, offering an elegant solution to efficient model adaptation. Instead of modifying the entire model, LoRA injects trainable rank decomposition matrices into the model's attention layers. This approach typically reduces trainable parameters by about 90% while maintaining comparable performance to full fine-tuning.

Here's a simple example of configuring LoRA:

```python
from peft import LoraConfig

config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,              # Rank of update matrices
    lora_alpha=32,    # Scale of updates
    lora_dropout=0.1  # Dropout probability
)
```

### [Prompt Tuning](./prompt_tuning.md)

Prompt tuning offers an even lighter approach by adding trainable tokens to the input rather than modifying model weights. This method is particularly effective for:

```python
from peft import PromptTuningConfig

config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=8,    # Number of trainable tokens
    prompt_tuning_init="TEXT", # Initialize from text
    tokenizer_name_or_path="your-base-model"
)
```

Prompt tuning excels in scenarios requiring:
- Minimal memory footprint
- Quick task switching without model reloading
- Multi-task scenarios with limited resources

## Implementation Guide

Getting started with PEFT is straightforward. First, install the required packages:

```bash
pip install transformers peft accelerate
```

Choose your PEFT method based on your requirements:
- For general use cases, LoRA provides an excellent balance of efficiency and performance
- For extremely limited resources or frequent task switching, consider prompt tuning
- When working with vision models or specialized architectures, explore adapter methods

Our `notebooks/` directory contains practical examples:
- `lora_finetuning.ipynb`: Complete LoRA implementation walkthrough
- `prompt_tuning_example.ipynb`: Guide to effective prompt tuning

## Best Practices

When implementing PEFT methods:
1. Start with smaller rank values (4-8) for LoRA and increase if needed
2. Monitor training loss carefully to avoid overfitting
3. Use validation sets to compare performance with full fine-tuning
4. Consider task requirements when choosing between methods

## Advanced Topics

PEFT methods can be combined with other optimization techniques:
- 8-bit quantization for further memory savings
- Gradient accumulation for larger effective batch sizes
- Multi-GPU training for faster iteration

## Resources
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)
- [Hugging Face PEFT Guide](https://huggingface.co/blog/peft)
