# VLM Fine-Tuning
## Efficient Fine-Tuning

### Quantization
Quantization reduces the precision of model weights and activations, significantly lowering memory usage and speeding up computations. For example, switching from `float32` to `bfloat16` halves memory requirements per parameter while maintaining performance. For more aggressive compression, 8-bit and 4-bit quantization can be used, further reducing memory usage, though at the cost of some accuracy. These techniques can be applied to both the model and optimizer settings, enabling efficient training on hardware with limited resources.

### PEFT & LoRA
As introduced in Module 3, LoRA (Low-Rank Adaptation) focuses on learning compact rank-decomposition matrices while keeping the original model weights frozen. This drastically reduces the number of trainable parameters, cutting resource requirements significantly. LoRA, when integrated with PEFT, enables fine-tuning of large models by only adjusting a small, trainable subset of parameters. This approach is particularly effective for task-specific adaptations, reducing billions of trainable parameters to just millions while maintaining performance.

### Batch Size Optimization
To optimize the batch size for fine-tuning, start with a large value and reduce it if out-of-memory (OOM) errors occur. Compensate by increasing `gradient_accumulation_steps`, effectively maintaining the total batch size over multiple updates. Additionally, enable `gradient_checkpointing` to lower memory usage by recomputing intermediate states during the backward pass, trading computation time for reduced activation memory requirements. These strategies maximize hardware utilization and help overcome memory constraints.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",  # Directory for model checkpoints
    per_device_train_batch_size=4,   # Batch size per device (GPU/TPU)
    num_train_epochs=3,              # Total training epochs
    learning_rate=5e-5,              # Learning rate
    save_steps=1000,                 # Save checkpoint every 1000 steps
    bf16=True,                       # Use mixed precision for training
    gradient_checkpointing=True,     # Enable to reduce activation memory usage
    gradient_accumulation_steps=16,  # Accumulate gradients over 16 steps
    logging_steps=50                 # Log metrics every 50 steps
)
```

## **Supervised Fine-Tuning (SFT)**

Supervised Fine-Tuning (SFT) adapts a pre-trained Vision Language Model (VLM) to specific tasks by leveraging labeled datasets containing paired inputs, such as images and corresponding text. This method enhances the model's ability to perform domain-specific or task-specific functions, such as visual question answering, image captioning, or chart interpretation.

### **Overview**
SFT is essential when you need a VLM to specialize in a particular domain or solve specific problems where the base model's general capabilities may fall short. For example, if the model struggles with unique visual features or domain-specific terminology, SFT allows it to focus on these areas by learning from labeled data.

While SFT is highly effective, it has notable limitations:
- **Data Dependency**: High-quality, labeled datasets tailored to the task are necessary.
- **Computational Resources**: Fine-tuning large VLMs is resource-intensive.
- **Risk of Overfitting**: Models can lose their generalization capabilities if fine-tuned too narrowly.

Despite these challenges, SFT remains a robust technique for enhancing model performance in specific contexts.


### **Usage**
1. **Data Preparation**: Start with a labeled dataset that pairs images with text, such as questions and answers. For example, in tasks like chart analysis, the dataset `HuggingFaceM4/ChartQA` includes chart images, queries, and concise responses.

2. **Model Setup**: Load a pre-trained VLM suitable for the task, such as `HuggingFaceTB/SmolVLM-Instruct`, and a processor for preparing text and image inputs. Configure the model for supervised learning and suitability for your hardware.

3. **Fine-Tuning Process**:
   - **Formatting Data**: Structure the dataset into a chatbot-like format, pairing system messages, user queries, and corresponding answers.
   - **Training Configuration**: Use tools like Hugging Face's `TrainingArguments` or TRL's `SFTConfig` to set up training parameters. These include batch size, learning rate, and gradient accumulation steps to optimize resource usage.
   - **Optimization Techniques**: Use **gradient checkpointing** to save memory during training. Use quantized model to reduce memory requirements and speed up computations.
   - Employ `SFTTrainer` trainer from the TRL library, to streamline the training process.




## Preference Optimization


## Resources

- https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl
- https://huggingface.co/blog/dpo_vlm

## Next Steps

⏩ Try the [vlm_finetune_sample.ipynb](./notebooks/vlm_finetune_sample.ipynb) to implement this unified approach to preference alignment.