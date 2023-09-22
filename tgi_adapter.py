from huggingface_hub import InferenceClient
import uuid
import time

from huggingface_hub.inference._text_generation import TextGenerationResponse, TextGenerationStreamResponse
from pydantic import BaseModel, validator
from functools import lru_cache
from typing import List, Union, Optional, Dict


class Message(BaseModel):
    role: str
    content: str

    @validator("role")
    def validate_role(cls, v):
        if v not in ["system", "user"]:
            raise ValueError("Role must be either 'system' or 'user'.")
        return v


class OpenAIRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[Union[float, None]] = 1.0
    top_p: Optional[Union[float, None]] = 1.0
    n: Optional[Union[int, None]] = 1
    stop: Optional[Union[str, List[str], None]] = None
    max_tokens: Optional[Union[int, None]] = 100
    presence_penalty: Optional[Union[float, None]] = 0.0
    frequency_penalty: Optional[Union[float, None]] = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    user: Optional[str] = None

    @validator("model", pre=True, always=True)
    def validate_model(cls, v):
        if v != "llama2-70b-chat":
            raise ValueError("Only 'llama2-70b-chat' is supported.")
        return v

    @validator('temperature', pre=True, always=True)
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError("temperature must be between 0 and 2")
        return v

    @validator('top_p', pre=True, always=True)
    def validate_top_p(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("top_p must be between 0 and 1")
        return v

    @validator('n', pre=True, always=True)
    def validate_n(cls, v):
        if v is not None and v < 1:
            raise ValueError("n must be greater than 0")
        return v

    @validator('presence_penalty', 'frequency_penalty', pre=True, always=True)
    def validate_penalties(cls, v):
        if v is not None and (v < -2.0 or v > 2.0):
            raise ValueError("penalties must be between -2.0 and 2.0")
        return v

    @validator('stop', pre=True, always=True)
    def validate_stop(cls, v):
        if isinstance(v, list) and len(v) > 4:
            raise ValueError("stop sequence list can have up to 4 sequences")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama2-70b-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Always add a joke after the response."
                    },
                    {
                        "role": "user",
                        "content": "Hello!"
                    }
                ],
                "stream": True
            }
        }


class TGIAdapter:
    def __init__(self, tgi_endpoint):
        self.client: InferenceClient = InferenceClient(tgi_endpoint)

    def tokenize(self, message: Message):
        return message.content.split()

    @lru_cache(maxsize=500)
    def openai_to_tgi_request(self, openai_request_str: str):
        # bring back the hashable string to object:
        openai_request = OpenAIRequest.model_validate_json(openai_request_str)

        messages = openai_request.messages
        prompt = "<s>"
        for i in range(len(messages)):
            if messages[i].role == "system":
                prompt += f"[INST] <<SYS>>\n{messages[i].content}\n<</SYS>>\n"
            else:
                prompt += f"{messages[i].content} [/INST] "
                if i > 0:
                    prompt += f"{messages[i - 1].content}"
            prompt += "</s><s>[INST] "
        prompt += "[/INST]"
        return prompt

    def tgi_to_openai_response(self, tgi_response: TextGenerationResponse, unique_id, timestamp, prompt_tokens_len):
        token_text = tgi_response.generated_text
        print(tgi_response)
        response = {
            "id": unique_id,
            "object": "chat.completion",
            "created": timestamp,
            "model": "meta-llama/Llama-2-70b-hf",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": token_text
                },
                "finish_reason": str(tgi_response.details.finish_reason.value) if tgi_response.details else None,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens_len,  # Assuming tgi_prompt is a string
                "completion_tokens": tgi_response.details.generated_tokens,
                "total_tokens": prompt_tokens_len + tgi_response.details.generated_tokens
            }
        }
        return response

    def process_request(self, openai_request: OpenAIRequest):
        tgi_prompt = self.openai_to_tgi_request(openai_request.model_dump_json())
        unique_id = "chatcmpl-" + str(uuid.uuid4())  # Prefixing with "chatcmpl-"
        timestamp = int(time.time())

        # Call the TGI API without streaming
        tgi_response = self.client.text_generation(
            tgi_prompt,
            max_new_tokens=openai_request.max_tokens,
            details=True
        )

        # Convert the TGI response to the OpenAI format
        openai_response = self.tgi_to_openai_response(
            tgi_response, unique_id, timestamp,
            len(self.tokenize(openai_request.messages[-1]))
        )

        return openai_response

    def tgi_to_openai_response_chunk(self, tgi_response: TextGenerationStreamResponse, unique_id, timestamp):
        token_text = tgi_response.token.text
        print(tgi_response.details)
        chunk = {
            "id": unique_id,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": "meta-llama/Llama-2-70b-hf",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": token_text
                },
                "finish_reason": str(tgi_response.details.finish_reason.value) if tgi_response.details else None,
            }],
        }
        return chunk

    def process_request_stream(self, openai_request: OpenAIRequest):
        tgi_prompt = self.openai_to_tgi_request(openai_request.model_dump_json())
        unique_id = "chatcmpl-" + str(uuid.uuid4())  # Prefixing with "chatcmpl-"
        timestamp = int(time.time())

        # Call the TGI API with streaming
        for tgi_response in self.client.text_generation(
                tgi_prompt, max_new_tokens=openai_request.max_tokens, details=True, stream=True
        ):
            yield self.tgi_to_openai_response_chunk(tgi_response, unique_id, timestamp)
