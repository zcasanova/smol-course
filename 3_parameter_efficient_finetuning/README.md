# Parameter-Efficient Fine-Tuning (PEFT)

As language models grow larger, traditional fine-tuning becomes increasingly challenging. A full fine-tuning of a 7B parameter model requires substantial GPU memory, makes storing separate model copies expensive, and risks catastrophic forgetting of the model's original capabilities. Parameter-efficient fine-tuning (PEFT) methods address these challenges by modifying only a small subset of model parameters while keeping most of the model frozen.

Traditional fine-tuning updates all model parameters during training, which becomes impractical for large models. PEFT methods introduce innovative approaches to adapt models using far fewer trainable parameters - often less than 1% of the original model size. This dramatic reduction in trainable parameters enables:

- Fine-tuning on consumer hardware with limited GPU memory
- Storing multiple task-specific adaptations efficiently
- Better generalization in low-data scenarios
- Faster training and iteration cycles

## Available Methods

### 1️⃣ LoRA (Low-Rank Adaptation)

LoRA has emerged as the most widely adopted PEFT method, offering an elegant solution to efficient model adaptation. Instead of modifying the entire model, LoRA injects trainable rank decomposition matrices into the model's attention layers. This approach typically reduces trainable parameters by about 90% while maintaining comparable performance to full fine-tuning. [LoRA (Low-Rank Adaptation)](./lora_adapters.md)
 
### 2️⃣ Prompt Tuning](./prompt_tuning.md)

Prompt tuning offers an even lighter approach by adding trainable tokens to the input rather than modifying model weights. [Prompt Tuning](./prompt_tuning.md)

## Resources
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)
- [Hugging Face PEFT Guide](https://huggingface.co/blog/peft)
