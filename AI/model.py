from typing import Any, Dict, List, Literal

from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from pydantic import BaseModel


class BatchMessage(BaseModel):
    role: Literal[
        "user",
        "assistant",
        "developer",
        "system",
    ]
    content: ResponseInputMessageContentListParam


class BatchText(BaseModel):
    format: Dict[str, Any]


class BatchBody(BaseModel):
    model: Literal[
        "gpt-4o",
        "gpt-4o-mini",
        "o3-mini",
        "o1",
    ]
    input: List[BatchMessage]
    max_output_tokens: int
    temperature: float
    text: BatchText

    def to_dict(self):
        return {
            "model": self.model,
            "input": [message.model_dump() for message in self.input],
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "text": self.text.model_dump(),
        }


class BatchRequest(BaseModel):
    custom_id: str
    method: Literal["POST"]
    url: Literal[
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/completions",
        "/v1/embeddings",
    ]
    body: BatchBody


MODELS = Literal[
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4",
    "o3-mini",
    "o1-mini",
    "davinci-002",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
]


MAX_TOKENS = {
    "gpt-4o": 16384,
    "gpt-4": 8000,
    "gpt-4o-mini": 16384,
    "o3-mini": 16384,
    "o1-mini": 16384,
    "davinci-002": 4097,
    "gpt-3.5-turbo-0125": 16384,
    "gpt-3.5-turbo-1106": 16384,
    "gpt-4o-audio": 16384,
    "gpt-4o-mini-audio": 16384,
    "gpt-4.1-nano": 128000,
}
