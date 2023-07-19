import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage, ChatResult, ChatGeneration
from typing import Optional, List, Dict, Any

class Chat_AutoModels(BaseChatModel):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    device: str = None
    gen_kwargs: dict = None

    def _llm_type(self) -> str:
        return "model"  # Replace "model" with the appropriate type for your model

    def __init__(
        self,
        model_id: str,
        device_map: str,
        gen_kwargs: dict,
        torch_dtype: Optional[Any] = torch.float16,
        trust_remote_code: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self.device = self.model.device
        self.gen_kwargs = gen_kwargs

    def get_prompt(self, messages: List[BaseMessage]) -> str:
        prompt = f"{messages[0].content} "
        for i, message in enumerate(messages[1:]):
            if isinstance(message, HumanMessage):
                prompt += f"USER: {message.content} "
            elif isinstance(message, AIMessage):
                prompt += f"ASSISTANT: {message.content} "
        prompt += f"ASSISTANT:"
        return prompt

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        prompt = self.get_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors='pt')

        outputs = self.model.generate(inputs.input_ids.to(self.device), **self.gen_kwargs)
        generated_text = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        ai_message = AIMessage(content=generated_text.strip())
        chat_result = ChatResult(generations=[ChatGeneration(message=ai_message)])
        return chat_result

    def _agenerate(self):
        return None
