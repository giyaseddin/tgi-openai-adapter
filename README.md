# OpenAI-like Text Generation Inference (TGI) served with FastAPI

This project provides a FastAPI-based replica for text generation using HuggingFace 🤗 text generation like OpenAI.

It allows you to generate stream text responses for your messages and serve a model of your choice.


## Getting Started

These instructions will help you set up and run the FastAPI application for text generation.

Let's make it possible to use [text-generation-inference](https://github.com/huggingface/text-generation-inference) just like [OpenAI](https://platform.openai.com/docs/api-reference/chat/create)


### Prerequisites

- Python 3.7+
- FastAPI
- Text generation inference

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/giyaseddin/tgi-openai-adapter.git
   cd tgi-openai-adapter
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Replace OpenAI's `base_url` with your deployed URL and start using for text generation. 🚀

### Usage

1. Run the FastAPI application:

   ```bash
   uvicorn main:app --reload
   ```
   
2. With Docker

   ```bash
   cp .env.example .env
   ```
Then update the environment variables

Build the Docker Images

   ```bash
   docker-compose build
   ```

Start the services

   ```bash
   docker-compose up -d
   ```

Once the services are up and running, you can access the application at the specified port. For example, if your application is running on port 8000
Open a web browser and go to http://localhost:8000/docs to see the OpenAPI Swagger docs.


3. Make POST requests to the `/v1/chat/completions` endpoint to generate text. You can use the following JSON format for your requests:

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

```

data: {"choices": [{"delta": {"content": " come"}, "finish_reason": null, "index": 0}], "created": 1695169455, "id": "5b67aed3-2980-4332-9677-7397b60d3b5f", "model": "meta-llama/Llama-2-70b-hf", "object": "chat.completion.chunk"}

data: {"choices": [{"delta": {"content": " on"}, "finish_reason": null, "index": 0}], "created": 1695169455, "id": "5b67aed3-2980-4332-9677-7397b60d3b5f", "model": "meta-llama/Llama-2-70b-hf", "object": "chat.completion.chunk"}

data: {"choices": [{"delta": {"content": ","}, "finish_reason": null, "index": 0}], "created": 1695169455, "id": "5b67aed3-2980-4332-9677-7397b60d3b5f", "model": "meta-llama/Llama-2-70b-hf", "object": "chat.completion.chunk"}

data: {"choices": [{"delta": {"content": " that"}, "finish_reason": "stop", "index": 0}], "created": 1695169455, "id": "5b67aed3-2980-4332-9677-7397b60d3b5f", "model": "meta-llama/Llama-2-70b-hf", "object": "chat.completion.chunk"}

data: [DONE]
```


### Customization

- You can customize the `generate_text` method inside the FastAPI endpoint to adapt it to your specific text generation needs.



## Next up:
* Add regular chat completion
* Implement embedding endpoint
* Implement memory management (With RAG optional)
* Got identical? let's develop even more fun stuff 😉 


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


Certainly! If your project is using the Apache License 2.0, you can update the "License" section in the README accordingly:

## License

This project is licensed under the [Apache License 2.0](LICENSE) - see the [LICENSE](LICENSE) file for details.
