# these functions build from Langchain


import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage, ChatResult, ChatGeneration
import re
from pydantic import Field, root_validator
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

from langchain.utils import get_from_dict_or_env, get_pydantic_field_names

class Chat_Llama(BaseChatModel):
    tokenizer: LlamaTokenizer = None
    model: LlamaForCausalLM = None
    device: str = None
    gen_kwargs: dict = None
    model_name: str = None  # Add this line

    llama_schema: Optional[Dict[str, Any]] = None

    ####
    def _completion_to_aimessage(self, completion):
        content = completion['choices'][0]['text']
        return AIMessage(content=content)  # directly create an AIMessage with the content string

   
    def clean_output(self, content):
        # Remove single quotes and strip whitespaces
        cleaned_content = content.replace('\'', '').strip()
        return cleaned_content


    def _llm_type(self) -> str:
        return "model"
    #####
    def __init__(self, model_path: str, device_map: str, low_cpu_mem_usage: bool, gen_kwargs: dict, max_new_tokens: int,
                 llama_schema: Optional[Dict[str, Any]] = None,
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
        self.gen_kwargs['max_new_tokens'] = max_new_tokens  # Add this line
        self.model_name = model_path  # Set the model_name attribute
        self.llama_schema = llama_schema  # Set the llama_schema attribute
    ###
    class Config:
        """Configuration for this pydantic object."""
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            """Build extra kwargs from additional params that were passed in."""
            all_required_field_names = get_pydantic_field_names(cls)
            extra = values.get("model_kwargs", {})
            for field_name in list(values):
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                if field_name not in all_required_field_names:
                    logger.warning(
                        f"""WARNING! {field_name} is not default parameter.
                        {field_name} was transferred to model_kwargs.
                        Please confirm that {field_name} is what you intended."""
                    )
                    extra[field_name] = values.pop(field_name)

            invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
            if invalid_model_kwargs:
                raise ValueError(
                    f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                    f"Instead they were passed in as part of `model_kwargs` parameter."
                )

            values["model_kwargs"] = extra
            return values

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
    def _generate(self, text: Union[str, List[BaseMessage]], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        if isinstance(text, str):
            messages = [HumanMessage(content=text)]
        else:
            messages = text

        generated_messages = []
        default_args = {
            "sentiment": "",
            "language": ""
        }

        for message in messages:
            prompt = self.get_prompt([message])
            inputs = self.tokenizer(prompt, return_tensors='pt')

            outputs = self.model.generate(input_ids=inputs.input_ids.to(self.device), **self.gen_kwargs)
            generated_output = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

            # Process the output through your function to create an AIMessage
            ai_message = self._completion_to_aimessage({'choices': [{'text': generated_output}]})

            if self.llama_schema:
                function_call = {"name": "some_function", "arguments": default_args.copy()}

                sentiment_match = re.search("Sentiment: (.*?)\n", generated_output, re.DOTALL)
                if sentiment_match:
                    function_call["arguments"]["sentiment"] = sentiment_match.group(1).strip()

                    if "Positive" in sentiment_match.group(1):
                        function_call["arguments"]["stars"] = random.choice([4, 5])
                    elif "Neutral" in sentiment_match.group(1):
                        function_call["arguments"]["stars"] = random.choice([2, 3])
                    elif "Negative" in sentiment_match.group(1):
                        function_call["arguments"]["stars"] = random.choice([1, 2])

                language_match = re.search("Language: (.*?)(\n|$)", generated_output, re.DOTALL)
                if language_match:
                    function_call["arguments"]["language"] = language_match.group(1).strip()

                # Check if the message content is a dict (JSON-like)
                if isinstance(ai_message.content, dict):
                    message_dict = {
                        "content": ai_message.content,
                        "role": "assistant",
                        "function_call": function_call,  # don't convert function_call to a string
                    }
                else:
                    message_dict = {
                        "content": ai_message.content,
                        "role": "assistant",
                    }

                generated_message = self._convert_dict_to_message(message_dict)
            else:
                generated_message = ai_message

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
