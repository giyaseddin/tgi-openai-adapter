from huggingface_hub import InferenceClient
import uuid
import time
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

    @validator("stream", pre=True, always=True)
    def validate_stream(cls, v):
        if not v:
            raise ValueError("Only streaming is supported.")
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
        schema_extra = {
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
        self.client = InferenceClient(tgi_endpoint)

    @lru_cache(maxsize=500)
    def openai_to_tgi_request(self, openai_request_str: str):
        # bring back the hashable string to object:
        openai_request = OpenAIRequest.parse_raw(openai_request_str)

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

    def tgi_to_openai_chunk(self, tgi_response, unique_id, timestamp):
        token_text = tgi_response.token.text
        chunk = {
            "choices": [{
                "delta": {
                    "content": token_text
                },
                "finish_reason": None if tgi_response.details is None else "stop",
                "index": 0
            }],
            "created": timestamp,
            "id": unique_id,
            "model": "meta-llama/Llama-2-70b-hf",
            "object": "chat.completion.chunk"
        }
        return chunk

    def process_request_stream(self, openai_request: OpenAIRequest):
        tgi_prompt = self.openai_to_tgi_request(openai_request.json())
        unique_id = str(uuid.uuid4())
        timestamp = int(time.time())
        for tgi_response in self.client.text_generation(
                tgi_prompt, max_new_tokens=openai_request.max_tokens, details=True, stream=True
        ):
            yield self.tgi_to_openai_chunk(tgi_response, unique_id, timestamp)
