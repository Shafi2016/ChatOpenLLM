# these functions build from Langchain

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage, ChatResult, ChatGeneration
import re

import string
import random
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)


class Chat_Llama(BaseChatModel):
    tokenizer: LlamaTokenizer = None
    model: LlamaForCausalLM = None
    device: str = None
    gen_kwargs: dict = None
    model_name: str = None  # Add this line

    llama_schema: Optional[Dict[str, Any]] = None
    ######
    def _llm_type(self) -> str:
        return "model"

    ####

    def __init__(self, model_path: str, device_map: str, low_cpu_mem_usage: bool, gen_kwargs: dict, llama_schema: Optional[Dict[str, Any]] = None,
                 load_in_4bit: bool=True, load_in_8bit: bool=True, torch_dtype: Optional[Any]=torch.float16):
        super().__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        self.device = self.model.device
        self.gen_kwargs = gen_kwargs
        self.model_name = model_path  # Set the model_name attribute
        self.llama_schema = llama_schema  # Set the llama_schema attribute


    def get_prompt(self, messages: List[BaseMessage]) -> str:
        prompt = f"{messages[0].content} "
        for i, message in enumerate(messages[1:]):
            if isinstance(message, HumanMessage):
                prompt += f"USER: {message.content} "
            elif isinstance(message, AIMessage):
                prompt += f"ASSISTANT: {message.content} "
            elif isinstance(message, ChatMessage):
                prompt += f"CHAT: {message.content} "
            elif isinstance(message, FunctionMessage):
                prompt += f"FUNCTION: {message.name} "
        prompt += f"ASSISTANT:"
        return prompt


    ####
    def _convert_dict_to_message(self, _dict: Mapping[str, Any]) -> BaseMessage:
        role = _dict["role"]
        if role == "user":
            return HumanMessage(content=_dict["content"])
        elif role == "assistant":
            content = _dict["content"] or ""
            if _dict.get("function_call"):
                function_call = _dict["function_call"]
                additional_kwargs = {"function_call": function_call}
            else:
                additional_kwargs = {}
            return AIMessage(content=content, additional_kwargs=additional_kwargs)
        elif role == "system":
            return SystemMessage(content=_dict["content"])
        elif role == "function":
            return FunctionMessage(content=_dict["content"], name=_dict["name"])
        else:
            return ChatMessage(content=_dict["content"], role=role)




    ###
    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
            if "function_call" in message.additional_kwargs:
                message_dict["function_call"] = message.additional_kwargs["function_call"]
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, FunctionMessage):
            message_dict = {
                "role": "function",
                "content": message.content,
                "name": message.name,
            }
        else:
            raise ValueError(f"Got unknown type {message}")
        if "name" in message.additional_kwargs:
            message_dict["name"] = message.additional_kwargs["name"]
        return message_dict



    def _import_tiktoken(self) -> Any:
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_token_ids. "
                "Please install it with `pip install tiktoken`."
            )
        return tiktoken
    ########
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        generated_messages = []
        default_args = {
            "sentiment": "",
            "language": ""
        }

        for message in messages:
            prompt = self.get_prompt([message])
            inputs = self.tokenizer(prompt, return_tensors='pt')

            outputs = self.model.generate(input_ids=inputs.input_ids.to(self.device), **self.gen_kwargs)
            generated_text = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

            #print(f"Generated Text: {generated_text}")  # Debugging print statement

            if self.llama_schema:
                function_call = {"name": "some_function", "arguments": default_args.copy()}

                sentiment_match = re.search("Sentiment: (.*?)\n", generated_text, re.DOTALL)
                if sentiment_match:
                    function_call["arguments"]["sentiment"] = sentiment_match.group(1).strip()

                    if "Positive" in sentiment_match.group(1):
                        function_call["arguments"]["stars"] = random.choice([4, 5])
                    elif "Neutral" in sentiment_match.group(1):
                        function_call["arguments"]["stars"] = random.choice([2, 3])
                    elif "Negative" in sentiment_match.group(1):
                        function_call["arguments"]["stars"] = random.choice([1, 2])

                language_match = re.search("Language: (.*?)(\n|$)", generated_text, re.DOTALL)
                if language_match:
                    function_call["arguments"]["language"] = language_match.group(1).strip()

                #print(f"Function Call: {function_call}")  # Debugging print statement

                message_dict = {
                    "content": "",
                    "role": "assistant",
                    "function_call": function_call,  # don't convert function_call to a string
                }

                generated_message = self._convert_dict_to_message(message_dict)
            else:
                generated_message = AIMessage(content=generated_text)

            generated_messages.append(ChatGeneration(message=generated_message))

        return ChatResult(generations=generated_messages)



    def _agenerate(self):
         raise NotImplementedError("Asynchronous generation is not supported.")

    ###

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {"token_usage": overall_token_usage}

