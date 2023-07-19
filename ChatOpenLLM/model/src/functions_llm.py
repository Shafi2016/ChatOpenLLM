from typing import Any, Dict, List, Union
from pydantic import BaseModel, root_validator
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import _convert_schema, get_llm_kwargs
from langchain.prompts import ChatPromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import (
    BaseLLMOutputParser,
    ChatGeneration,
    Generation,
    OutputParserException,
)
import json


class OutputFunctionsParser(BaseLLMOutputParser[Any]):
    args_only: bool = True

    def parse_result(self, result: List[Generation]) -> Any:
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message
        func_call = message.additional_kwargs.get("function_call")
        if func_call is None:
            raise OutputParserException("No function call found in the message")

        if isinstance(func_call, str):
            try:
                func_call = json.loads(func_call)
            except ValueError as exc:
                raise OutputParserException(f"Could not parse function call: {exc}")

        if not isinstance(func_call, dict):
            raise OutputParserException("Function call is not a dictionary")

        # Check if the function call includes both a name and arguments
        if 'name' not in func_call or 'arguments' not in func_call:
            raise OutputParserException("Function call does not include a name and arguments")

        args = func_call.get("arguments")
        if args is None:
            raise OutputParserException("No arguments found in the function call")

        return args

class JsonOutputFunctionsParser2(OutputFunctionsParser):
    def parse_result(self, result: List[Generation]) -> Any:
        func = super().parse_result(result)

        if isinstance(func, str):
            try:
                func = json.loads(func)
            except ValueError:
                raise OutputParserException("Could not parse func as JSON")

        if not isinstance(func, dict):
            raise OutputParserException("Func is not a dictionary")

        # Directly return func
        return func



def _get_tagging_function(schema: dict) -> dict:
    return {
        "name": "information_extraction",
        "description": "Extracts the relevant information from the passage.",
        "parameters": _convert_schema(schema),
    }

_TAGGING_TEMPLATE = """
Given the following review, extract and classify the following information:
1. The sentiment of the review, which can be "positive", "neutral", or "negative".
2. The number of stars given by the reviewer, which can be an integer between 1 and 5.
3. The language in which the review is written, which can be "Spanish", "English", "French", "German", or "Italian".

Passage:
{input}
"""


def create_tagging_chain2(schema: dict, llm: BaseLanguageModel) -> Chain:
    function = _get_tagging_function(schema)
    prompt = ChatPromptTemplate.from_template(_TAGGING_TEMPLATE)
    output_parser = JsonOutputFunctionsParser2()
    llm_kwargs = get_llm_kwargs(function)
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
    )
    return chain