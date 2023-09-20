# OpenAI-like Text Generation Inference (TGI) served with FastAPI

This project provides a FastAPI-based endpoint for text generation using a text generation adapter. It allows you to generate multiple text responses for a list of prompts using different text generation models.

## Getting Started

These instructions will help you set up and run the FastAPI application for text generation.

### Prerequisites

- Python 3.7+
- FastAPI
- Your text generation adapter

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/giyaseddin/tgi-openai-adapter.git
   cd tgi-openai-adapter
   ```

2. Install dependencies:

   ```bash
   pip install fastapi
   ```

3. Replace `your_adapter_module` and `AdapterClass` with the actual module and class you're using for text generation.

### Usage

1. Run the FastAPI application:

   ```bash
   uvicorn main:app --reload
   ```

2. Make POST requests to the `/v1/chat/completions` endpoint to generate text. You can use the following JSON format for your requests:

   ```json
   {
      "model": "llama2-70b-chat",
      "messages": [
        {
          "role": "system",
          "content": "You are a funny assistant. Always add a joke after the response."
        },
        {
          "role": "user",
          "content": "Hello!"
        }
      ],
      "stream": true
    }
   ```
   
following the [docs of OpenAI](https://platform.openai.com/docs/api-reference/chat/create)

   - `models`: A list of text generation model names.
   - `messages`: A list of messages consisted of  for text generation.
   - `max_tokens` (optional): Maximum number of tokens to generate per prompt (default is 50).
   - ... etc.

3. The endpoint will return a response identical to what OpenAI produces

#### Example output

Here's an example of a generation request:

```json

data: {"choices": [{"delta": {"content": " come"}, "finish_reason": null, "index": 0}], "created": 1695169455, "id": "5b67aed3-2980-4332-9677-7397b60d3b5f", "model": "meta-llama/Llama-2-70b-hf", "object": "chat.completion.chunk"}

data: {"choices": [{"delta": {"content": " on"}, "finish_reason": null, "index": 0}], "created": 1695169455, "id": "5b67aed3-2980-4332-9677-7397b60d3b5f", "model": "meta-llama/Llama-2-70b-hf", "object": "chat.completion.chunk"}

data: {"choices": [{"delta": {"content": ","}, "finish_reason": null, "index": 0}], "created": 1695169455, "id": "5b67aed3-2980-4332-9677-7397b60d3b5f", "model": "meta-llama/Llama-2-70b-hf", "object": "chat.completion.chunk"}

data: {"choices": [{"delta": {"content": " that"}, "finish_reason": "stop", "index": 0}], "created": 1695169455, "id": "5b67aed3-2980-4332-9677-7397b60d3b5f", "model": "meta-llama/Llama-2-70b-hf", "object": "chat.completion.chunk"}

data: [DONE]
```


### Customization

- You can customize the `generate_text` method inside the FastAPI endpoint to adapt it to your specific text generation needs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


Certainly! If your project is using the Apache License 2.0, you can update the "License" section in the README accordingly:

## License

This project is licensed under the [Apache License 2.0](LICENSE) - see the [LICENSE](LICENSE) file for details.
