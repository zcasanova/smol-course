# a smol course

![smolcourse image](./banner.png)

This is a practical course on aligning smol language models for your specific use case. It's a handy way to get started with aligning language models, because everything runs on most local machines, there's minimal GPU requirements, and no paid services. The course is based on the [SmolLM2](https://github.com/huggingface/smollm/tree/main) series of models, but you can transfer the skills you learn here to larger models or other small language models.

## Course Outline

This course provides a practical, hands-on approach to working with small language models, from initial training through to production deployment.

| Module | Description | Status | Release Date |
|--------|-------------|---------|--------------|
| [Instruction Tuning](./1_instruction_tuning) | Learn supervised fine-tuning, chat templating, and basic instruction following | ✅ Complete | Dec 3, 2024 |
| [Preference Alignment](./2_preference_alignment) | Explore DPO and ORPO techniques for aligning models with human preferences | 🚧 In Progress  | Dec 6, 2024 |
| [Parameter-efficient Fine-tuning](./3_parameter_efficient_finetuning) | Learn LoRA, prompt tuning, and efficient adaptation methods | 🚧 In Progress | Dec 9, 2024 |
| [Evaluation](./4_evaluation) | Use automatic benchmarks and create custom domain evaluations | 🚧 In Progress | Dec 16, 2024 |
| [Vision-language Models](./5_vision_language_models) | Adapt multimodal models for vision-language tasks | 📝 Planned | Dec 20, 2024 |
| [Synthetic Datasets](./6_synthetic_datasets) | Create and validate synthetic datasets for training | 📝 Planned | Dec 23, 2024 |
| [Inference](./7_inference) | Infer with models efficiently | 📝 Planned | Dec 27, 2024 |
| [Deployment](./8_deplyment) | Deploy and serve models at scale | 📝 Planned | Dec 30, 2024 |

## Why Small Language Models?

While large language models have shown impressive capabilities, they often require significant computational resources and can be overkill for focused applications. Small language models offer several advantages for domain-specific applications:

- **Efficiency**: Require significantly less computational resources to train and deploy
- **Customization**: Easier to fine-tune and adapt to specific domains
- **Control**: Better understanding and control of model behavior
- **Cost**: Lower operational costs for training and inference
- **Privacy**: Can be run locally without sending data to external APIs

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
