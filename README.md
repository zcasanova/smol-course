# smol-course

A practical guide to using smol language models (SLMs) for your own projects.

The smol-course teaches you how to adapt the smol series of language models for your specific domain or use case. While large language models   have shown impressive capabilities, they often require significant computational resources and can be overkill for focused applications. This course shows you how to:

- Fine-tune smaller, more efficient models for your specific needs
- Align model outputs with domain requirements and human preferences  
- Evaluate and improve model performance for your use case
- Create synthetic datasets for training and evaluation

## Course Outline

| Module | Description | Status |
|--------|-------------|--------|
| [Instruction Tuning](./1_instruction_tuning) | Learn how to fine-tune language models on instruction datasets using supervised learning techniques | ✅ Complete |
| [Preference Alignment](./2_preference_alignment) | Explore techniques for aligning language models with human preferences using reward modeling and RLHF | 🚧 In Progress |
| [Parameter-efficient Fine-tuning](./3_parameter_efficient_finetuning) | Learn efficient methods for adapting large language models with minimal computational resources | 📝 Planned |
| [Evaluation](./4_evaluation) | Understand how to evaluate language models using benchmarks and custom domain-specific metrics | ✅ Complete |
| [Vision-language Models](./5_vision_language_models) | Explore techniques for aligning multimodal models that work with both text and images | 📝 Planned |
| [Synthetic Datasets](./6_synthetic_datasets) | Learn how to create and use synthetic datasets for model training and evaluation | 📝 Planned |

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