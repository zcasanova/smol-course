# Chat Templates

Chat templates are essential for structuring interactions between language models and users. They provide a consistent format for conversations, ensuring that models understand the context and role of each message while maintaining appropriate response patterns.

## Understanding Chat Templates

At their core, chat templates define how conversations should be formatted when communicating with a language model. They include system-level instructions, user messages, and assistant responses in a structured format that the model can understand. This structure helps maintain consistency across interactions and ensures the model responds appropriately to different types of inputs.

Here's a basic example of a chat template structure:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."}
]
```

## System Messages

System messages set the foundation for how the model should behave. They act as persistent instructions that influence all subsequent interactions. For example:

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

## Conversations

Chat templates maintain context through conversation history, storing previous exchanges between users and the assistant. This allows for more coherent multi-turn conversations:

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

## Implementation with Transformers

The transformers library provides built-in support for chat templates. Here's how to use them:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list"},
]

# Apply the chat template
formatted_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

## Custom Formatting
You can customize how different message types are formatted. For example, adding special tokens or formatting for different roles:

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
"""
```

## Multi-Turn Support

Templates can handle complex multi-turn conversations while maintaining context:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```


## Resources
- [Hugging Face Chat Templating Guide](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Chat Templates Examples Repository](https://huggingface.co/spaces/huggingface-projects/chat-templates) 