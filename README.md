# a smol course

![smolcourse image](./banner.png)


This is a practical and concice guide on aligning smol language models for your specific use case. It's a handy way to get started with aligning language models, because everything runs on most local machines. There's minimal GPU required and no paid services.

The smol-course teaches you how to adapt the smol series of language models for your specific domain or use case. While large language models  have shown impressive capabilities, they often require significant computational resources and can be overkill for focused applications. This course shows you how to:

- Fine-tune smaller, more efficient models for your specific needs
- Align model outputs with domain requirements and human preferences  
- Evaluate and improve model performance for your use case
- Create synthetic datasets for training and evaluation

You can transfer the skills you learn here to larger models and models other than SmolLM2 and SmolVLM, but this course is designed to run on those models.

## Course Outline

| Module | Description | Status | Release Date |
|--------|-------------|---------|--------------|
| [Instruction Tuning](./1_instruction_tuning) | Learn supervised fine-tuning, chat templating, and basic instruction following | ✅ Complete | Mar 12, 2024 |
| [Preference Alignment](./2_preference_alignment) | Explore DPO and ORPO techniques for aligning models with human preferences | ✅ Complete | Mar 14, 2024 |
| [Parameter-efficient Fine-tuning](./3_parameter_efficient_finetuning) | Learn LoRA, prompt tuning, and efficient adaptation methods | 🚧 In Progress | Mar 16, 2024 |
| [Evaluation](./4_evaluation) | Use automatic benchmarks and create custom domain evaluations | ✅ Complete | Mar 18, 2024 |
| [Vision-language Models](./5_vision_language_models) | Adapt multimodal models for vision-language tasks | 📝 Planned | Mar 20, 2024 |
| [Synthetic Datasets](./6_synthetic_datasets) | Create and validate synthetic datasets for training | 📝 Planned | Mar 22, 2024 |
| [Inference](./7_inference) | Deploy and serve models efficiently | 📝 Planned | Mar 24, 2024 |

### Why Small Language Models?

Small language models offer several advantages for domain-specific applications:

- **Efficiency**: Require significantly less computational resources to train and deploy
- **Customization**: Easier to fine-tune and adapt to specific domains
- **Control**: Better understanding and control of model behavior
- **Cost**: Lower operational costs for training and inference
- **Privacy**: Can be run locally without sending data to external APIs

This course provides a practical, hands-on approach to working with small language models, from initial training through to production deployment.

## Prerequisites

Before starting, ensure you have the following:
- Basic understanding of machine learning and natural language processing.
- Familiarity with Python, PyTorch, and the `transformers` library.
- Access to a pre-trained language model and a labeled dataset.

## Installation

All the examples run in the same environment, so you only need to install the dependencies once like this:

```bash
pip install -r requirements.txt
```
