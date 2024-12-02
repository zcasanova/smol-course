# Instruction Tuning

This tutorial provides a concise guide on instruction tuning of language models, focusing on the key concepts and steps involved in the process.

Instruction tuning involves adapting pre-trained models to specific tasks by further training them on task-specific datasets. This process helps models improve their performance on targeted tasks. Instruction tuning uses two primary formats to guide model interactions:

- **Instruction Format**: Provides explicit tasks for the model to perform.
- **Conversational Format**: Structures interactions as a dialogue between a user and a system.

## 1️⃣ Chat Templates

Chat templates structure interactions between users and AI models, ensuring consistent and contextually appropriate responses. They include components like system prompts and role-based messages.

For more detailed information, refer to the [Chat Templates](./chat_templates.md) section.

## 2️⃣ Supervised Fine-Tuning

Supervised Fine-Tuning (SFT) is a critical process for adapting pre-trained language models to specific tasks. It involves training the model on a task-specific dataset with labeled examples.

For a detailed guide on SFT, including key steps and best practices, see the [Supervised Fine-Tuning](./supervised_fine_tuning.md) page.


## Example Notebooks

- [Chat Templates](./notebooks/chat_templates_example.ipynb)
- [Supervised Fine-Tuning](./notebooks/supervised_fine_tuning_tutorial.ipynb)

## References

- [Transformers documentation on chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Script for Supervised Fine-Tuning in TRL]([./scripts/supervised_finetuning.py](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py))
- [`SFTTrainer` in TRL]([./chat_templates.md](https://huggingface.co/docs/trl/main/en/sft_trainer))
- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
- [Supervised Fine-Tuning with TRL](https://huggingface.co/docs/trl/main/en/tutorials/supervised_finetuning)
- [How to fine-tune Google Gemma with ChatML and Hugging Face TRL](https://www.philschmid.de/fine-tune-google-gemma)
- [Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)