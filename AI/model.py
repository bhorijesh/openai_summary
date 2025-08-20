from typing import Literal

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
