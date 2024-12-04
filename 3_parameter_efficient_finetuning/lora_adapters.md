# LoRA (Low-Rank Adaptation)

LoRA has become the most widely adopted PEFT method. It works by adding small rank decomposition matrices to the attention weights, typically reducing trainable parameters by about 90%. Here's a basic configuration:

```python
from peft import LoraConfig

config = LoraConfig(
    task_type="CAUSAL_LM",  # Task type for the model
    r=8,  # Rank of update matrices
    lora_alpha=32,  # Scale of updates
    lora_dropout=0.1,  # Dropout probability for LoRA layers
)
```

This configuration can be used with TRL to train a model with LoRA. LoRA matrices are known as adapters and can be merged back into the base model.

## Merging LoRA Adapters

After training with LoRA, you might want to merge the adapter weights back into the base model for easier deployment. This creates a single model with the combined weights, eliminating the need to load adapters separately during inference.

The merging process requires attention to memory management and precision. Since you'll need to load both the base model and adapter weights simultaneously, ensure sufficient GPU/CPU memory is available. Using `device_map="auto"` can help with automatic memory management. Maintain consistent precision (e.g., float16) throughout the process, matching the precision used during training and saving the merged model in the same format for deployment. Before deploying, always validate the merged model by comparing its outputs and performance metrics with the adapter-based version.

### Memory Requirements and Precautions

1. Ensure at least 2x the base model size in available memory <br>
2. For models >10B parameters, consider using device_map="sequential" <br>
3. Monitor GPU memory usage during merging process <br>
4. Keep consistent precision throughout the process <br>

### Basic Merging Process

Here's how to merge a LoRA adapter back into the base model:

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Load the PEFT model with adapter
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/adapter",
    torch_dtype=torch.float16
)

# 3. Merge adapter weights with base model
try:
    merged_model = peft_model.merge_and_unload()
except RuntimeError as e:
    print(f"Merging failed: {e}")
    # Implement fallback strategy or memory optimization

# 4. Save the merged model
merged_model.save_pretrained("path/to/save/merged_model")
```

If you encounter size discrepancies in the saved model, ensure you're also saving the tokenizer:

```python
# Save both model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("base_model_name")
merged_model.save_pretrained("path/to/save/merged_model")
tokenizer.save_pretrained("path/to/save/merged_model")
```

### Multiple Adapter Composition

You can combine multiple LoRA adapters:

```python
from peft import PeftModel

def merge_adapters(base_model, adapter_paths, weights=None):
    """
    Merge multiple adapters with optional weighting
    """
    if weights is None:
        weights = [1.0] * len(adapter_paths)
    
    current_model = base_model
    for path, weight in zip(adapter_paths, weights):
        adapter = PeftModel.from_pretrained(current_model, path)
        current_model = adapter.merge_and_unload(adapter_weight=weight)
    
    return current_model
```

## Recommended Hyperparameters

Task-specific recommendations:

1. **Text Classification:**

- r: 4-8 <br>
- lora_alpha: 16-32 <br>
- lora_dropout: 0.05-0.1 <br>


2. **Generation Tasks:**

- r: 8-32 <br>
- lora_alpha: 32-64 <br>
- lora_dropout: 0.1-0.2 <br>


3. **Language Translation:**

- r: 16-64 <br>
- lora_alpha: 32-128 <br>
- lora_dropout: 0.1-0.15 <br>

## Implementation Guide

Start by installing the required packages:
```bash
pip install transformers peft accelerate
```

The `notebooks/` directory contains practical tutorials for implementing different PEFT methods. Begin with `lora_finetuning.ipynb` for a basic introduction, then explore prompt and prefix tuning through their respective notebooks.

When implementing PEFT methods, start with small rank values (4-8) for LoRA and monitor training loss. Use validation sets to prevent overfitting and compare results with full fine-tuning baselines when possible. The effectiveness of different methods can vary by task, so experimentation is key.

## Using TRL with PEFT

PEFT methods can be combined with TRL (Transformer Reinforcement Learning) for efficient reinforcement learning fine-tuning. This integration is particularly useful for RLHF (Reinforcement Learning from Human Feedback) as it reduces memory requirements.

You can scale training across multiple GPUs while keeping memory usage efficient. Here's how to set it up:

```python
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM

# Get current device from accelerator
accelerator = Accelerator()
current_device = accelerator.process_index

# Load model with PEFT config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load model on specific device
model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    load_in_8bit=True,  # Optional: use 8-bit precision
    device_map={"": current_device},  # Assign to correct device
    peft_config=lora_config
)
```

# Resources


1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
2. Prefix Tuning: [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
3. Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) 
4. P-Tuning: [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf) 
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Hugging Face PEFT Guide](https://huggingface.co/blog/peft)
