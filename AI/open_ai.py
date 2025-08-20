import json
import logging
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

from AI.model import MAX_TOKENS, MODELS

# Configure logging
logger = logging.getLogger("ai")


class OpenAIModel:
    """
    A flexible class to handle interactions with OpenAI's API.
    Provides customizable methods for text generation, function calling, and response parsing.
    """

    def __init__(
        self,
        model_name: MODELS,
        messages: List[Dict[str, Any]],
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
        store_logs: bool = False,
        stream: bool = False,
    ):
        """Initialize OpenAI model with flexible configuration parameters"""
        # Initialize OpenAI client with configuration from environment
        self.__client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            max_retries=3,
            timeout=60,  # 1-minute timeout
        )
        
        # Core parameters
        self.model_name = model_name
        self.messages = [messages] if isinstance(messages, dict) else messages
        self.temperature = temperature
        self.max_tokens = max_tokens or MAX_TOKENS.get(model_name, 1000)
        self.store = store_logs
        self.stream = stream

        # Tool calling state
        self.tool_call_loop = 0
        self.tool_call_limit = None
        self.valid_tools = False
        self.recall = False

        # Storage
        self.__usages = []
        self.response = None

        # Validate initial configuration
        self.initial_validation()

    @classmethod
    def from_message(
        cls,
        model_name: MODELS,
        system_message: str,
        user_message: str,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
        store_logs: bool = False,
        stream: bool = False,
    ) -> "OpenAIModel":
        """
        Create an instance of OpenAIModel from a user message.
        """
        # Format the system and user messages
        formatted_messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user", 
                "content": user_message,
            },
        ]
        return cls(
            model_name=model_name,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            store_logs=store_logs,
            stream=stream,
        )

    def add_message(self, message: Dict[str, Any]) -> None:
        """
        Add a message to the conversation history.
        """
        self.messages.append(message)

    @property
    def client(self) -> OpenAI:
        """Get OpenAI client instance"""
        return self.__client

    @property
    def usages(self) -> List[Dict[str, Any]]:
        """Get usage information for each call"""
        return self.__usages

    @property
    def body(self) -> Dict[str, Any]:
        """Build request body with current configuration"""
        self.validate_messages()
        body = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        return body

    def validate_tools(
        self, tools: Dict[str, Callable], tool_schemas: List[Dict[str, Any]]
    ) -> None:
        """Validate that all tool schemas have corresponding callable functions"""
        for tool in tool_schemas:
            tool_name = tool["function"]["name"]
            if tool_name not in tools:
                logger.error(
                    "[VALIDATE TOOLS] TOOL NOT FOUND: %s",
                    tool_name,
                    tools,
                    tool_schemas,
                )
                raise ValueError(f"Tool {tool_name} not found in tools")
            if not callable(tools[tool_name]):
                raise ValueError(f"Tool {tool_name} is not callable")
        self.valid_tools = True

    def initial_validation(self) -> None:
        """Validate initial configuration parameters"""
        if not 0 <= self.temperature <= 2:
            logger.error(
                "[INITIAL VALIDATION] TEMPERATURE OUT OF RANGE: %s",
                self.temperature,
            )
            raise ValueError("Temperature must be between 0 and 2")

        if not 1 <= self.max_tokens <= MAX_TOKENS[self.model_name]:
            logger.error(
                "[INITIAL VALIDATION] MAX TOKENS OUT OF RANGE: %s",
                self.max_tokens,
                MAX_TOKENS[self.model_name],
            )
            raise ValueError(
                f"Max tokens must be between 1 and {MAX_TOKENS[self.model_name]}"
            )

    def validate_messages(self) -> None:
        """Validate that the messages are in the correct format"""
        user_message_exists = False

        for message in self.messages:
            if message.get("role") == "user":
                user_message_exists = True

        if not user_message_exists:
            logger.error(
                "[VALIDATE MESSAGES] USER MESSAGE NOT FOUND: %s", self.messages
            )
            raise ValueError("User message not found")

    def execute_text_response(self) -> Any:
        """Generate a text response with optional tool calling"""
        self.response = self.client.chat.completions.create(**self.body)
        self.__usages.append(self.response.usage.model_dump())
        logger.info(
            "[TEXT RESPONSE] RESPONSE:\n %s \n USAGE:\n %s",
            self.response.choices[0].message.content,
            self.__usages,
        )

        return self.response

    def execute_parsed_response(self, structure: BaseModel) -> Any:
        """Generate a response parsed according to provided structure"""
        body = self.body.copy()
        body["response_format"] = {"type": "json_object"}
        self.response = self.client.chat.completions.create(**body)
        self.__usages.append(self.response.usage.model_dump())
        logger.info(
            "[PARSED RESPONSE] RESPONSE:\n %s \n USAGE:\n %s",
            self.response.choices[0].message.content,
            self.__usages,
        )
        return self.response

    def run_tool_calls(
        self,
        tools: Dict[str, Callable],
        tool_schemas: List[Dict[str, Any]],
        remove_called_tools: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute tool calls based on model response"""
        # Validate tools if not already validated
        if not self.valid_tools:
            self.validate_tools(tools, tool_schemas)

        # Set tool call limit if not set
        if self.tool_call_limit is None:
            self.tool_call_limit = len(tool_schemas)

        # Check for loop limit
        if self.tool_call_loop >= self.tool_call_limit:
            logger.error("Tool call loop limit reached. messages: %s", self.messages)
            raise Exception("Tool call loop limit reached")

        # Get response from OpenAI
        body = self.body.copy()
        body["tools"] = tool_schemas
        self.response = self.client.chat.completions.create(**body)
        self.__usages.append(
            {f"call_{self.tool_call_loop}": self.response.usage.model_dump()}
        )

        # Process tool calls
        if self.response.choices[0].message.tool_calls:
            for tool_call in self.response.choices[0].message.tool_calls:
                self.recall = True
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                result = tools[function_name](**arguments)

                # Store tool call results
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(result),
                    }
                )

                # Remove used tools if specified
                if remove_called_tools:
                    tool_schemas = [t for t in tool_schemas if t["function"]["name"] != function_name]
        else:
            self.recall = False

        self.tool_call_loop += 1

        # Recursively continue tool calls if needed
        if self.recall and tool_schemas:
            return self.run_tool_calls(tools, tool_schemas)

        # Return final messages if no more tool calls needed
        self.messages.append(
            {"role": "assistant", "content": self.response.choices[0].message.content}
        )
        return self.messages

    def extract_parsed_response(self, to_dict: bool = True) -> Any:
        """Get parsed response output"""
        if not self.response:
            logger.error(
                "[EXTRACT PARSED RESPONSE] NO RESPONSE AVAILABLE: %s",
                self.response,
            )
            raise ValueError("No response available. Call execute_text_response() first.")
        
        content = self.response.choices[0].message.content
        if to_dict:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"content": content}
        return content

    def extract_text_response(self) -> str:
        """Get raw text response output"""
        if not self.response:
            raise ValueError("No response available. Call execute_text_response() first.")
        return self.response.choices[0].message.content
