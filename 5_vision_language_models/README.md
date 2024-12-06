# Vision Language Models

## 1. VLM Usage

Vision Language Models (VLMs) are advanced AI models that combine vision and language understanding. They enable tasks such as image captioning, visual question answering, and multimodal reasoning by integrating visual and textual information.  

These models leverage pre-trained architectures to understand visual content and generate or process text based on it, enabling a wide array of applications across domains like healthcare, autonomous driving, and multimedia search.  

For more on the technicality and usage of VLMs, refer to the [VLM Usage](./vlm_usage.md) page.  


## 2. VLM Fine-Tuning  

Fine-tuning a VLM involves adapting a pre-trained model to perform specific tasks or to operate effectively on a particular dataset. This process customizes the model's capabilities while leveraging the knowledge it has acquired during pre-training. For a detailed guide on fine-tuning for VLMs, see the [VLM Fine-Tuning](./vlm_finetuning.md) page.


## Exercise Notebooks  


| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| VLM Usage | Learn how to load and use a pre-trained VLM for various tasks | 🐢 Process an image<br>🐕 Process multiple images with batch handling <br>🦁 Process a full video| [Notebook](./notebooks/vlm_usage_sample.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/user/project/vlm_usage_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| VLM Fine-Tuning | Learn how to fine-tune a pre-trained VLM for task-specific datasets | 🐢 Use a basic dataset for fine-tuning<br>🐕 Try a new dataset<br>🦁 Experiment with alternative fine-tuning methods | [Notebook](./notebooks/vlm_finetune_sample.ipynb)| <a target="_blank" href="https://colab.research.google.com/github/user/project/vlm_finetune_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 


## References  
- [Hugging Face: Fine-Tuning VLMs](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)  
- [Hugging Face: SmolVLM](https://huggingface.co/blog/smolvlm)  
- [Hugging Face: Vision Language Models](https://huggingface.co/blog/vlms)
- [CLIP: Learning Transferable Visual Models from Natural Language Supervision](https://arxiv.org/abs/2103.00020)  
- [Align Before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651)  

