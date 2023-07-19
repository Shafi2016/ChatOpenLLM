# ChatOpenLLM

ChatOpenLLM is an open-source Python package that provides functionality similar to the following OpenAI chat model for the open-source models:

```python
chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model='gpt-3.5-turbo'
)
```

Our package wraps the model loading and text generation functionalities into an accessible API, making it easy to create engaging dialogues with your AI models. ChatOpenLLM allows you to generate AI responses to human inputs, similar to the renowned ChatGPT, enabling you to harness the power of advanced AI models in your chat applications.

## Installation

You can install ChatOpenLLM via pip:

```
!git clone https://github.com/Shafi2016/ChatOpenLLM.git
%cd /content/ChatOpenLLM
!pip install .
%cd ..
```

### Example 1

Here's an example of how you can use OpenLLM:

```python
Any model with tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=True)
model = LlamaForCausalLM.from_pretrained(model_path) can be used with Chat_Llama()
from ChatOpenLLM import Chat_Llama, ChatGPTQ, Chat_AutoModels, create_tagging_chain2
llm = Chat_Llama("TheBloke/wizardLM-7B-HF", 
                 device_map='auto', 
                 low_cpu_mem_usage=True, 
                 load_in_4bit=True, 
                 gen_kwargs=dict(temperature=0))

```
## Example 1.1 : OpenAI-like Function for the Open Source LLM

```python
schema = {
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "stars": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "describes the number of stars give by a reviewer on Amazon",
        },
        "language": {
            "type": "string",
            "enum": ["spanish", "english", "french", "german", "italian"],
        },
    },
    "required": ["language", "sentiment", "stars"],
}

from ChatOpenLLM import Chat_Llama, ChatGPTQ, Chat_AutoModels, create_tagging_chain2

llm = Chat_Llama("TheBloke/Wizard-Vicuna-13B-Uncensored-HF",
                 device_map='auto',
                 low_cpu_mem_usage=True,
                 load_in_4bit=True,
                 gen_kwargs=dict(max_length=2048, temperature=0), 
                 llama_schema= schema)  # pass the llama_schema


chain = create_tagging_chain2(schema, llm)
inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
chain.run(inp)
{'sentiment': 'Positive', 'language': 'Spanish', 'stars': 4}

```
## The following have been tested with Chat_Llama()
| Serial Number | model_path |
| ------------- | ---------- |
| 1 | 'TheBloke/Wizard-Vicuna-13B-Uncensored-HF' |
| 2 | 'tiiuae/falcon-7b-instruct' |
| 3 | 'TheBloke/wizardLM-7B-HF' |

### Example 2
```python
You can initialize the `ChatGPTQ` class as follows:
Any model with tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path) could be used with function.
from ChatOpenLLM import Chat_Llama, ChatGPTQ, Chat_AutoModels, create_tagging_chain2
llm = ChatGPTQ(
    model_name_or_path="TheBloke/WizardLM-13B-V1-0-Uncensored-SuperHOT-8K-GPTQ",
    model_basename="wizardlm-13b-v1.0-uncensored-superhot-8k-GPTQ-4bit-128g.no-act.order",
    gen_kwargs=dict(max_length=4048, temperature=0, top_p=0.95, repetition_penalty=1.15),
    use_safetensors=True, 
    load_in_4bit=True, 
    trust_remote_code=True,
    device_map='auto',
    use_triton=False, 
    quantize_config=None,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
```

## Example 2.1 : OpenAI-like Function for the Open Source LLM

```python
schema = {
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "stars": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "describes the number of stars give by a reviewer on Amazon",
        },
        "language": {
            "type": "string",
            "enum": ["spanish", "english", "french", "german", "italian"],
        },
    },
    "required": ["language", "sentiment", "stars"],
}

from ChatOpenLLM import Chat_Llama, ChatGPTQ, Chat_AutoModels, create_tagging_chain2
llm = ChatGPTQ(
    model_name_or_path="TheBloke/WizardLM-13B-V1-0-Uncensored-SuperHOT-8K-GPTQ",
    model_basename="wizardlm-13b-v1.0-uncensored-superhot-8k-GPTQ-4bit-128g.no-act.order",
    gen_kwargs=dict(max_length=4048, temperature=0, top_p=0.95, repetition_penalty=1.15),
    use_safetensors=True, 
    load_in_4bit=True, 
    trust_remote_code=True,
    device_map='auto',
    use_triton=False, 
    quantize_config=None,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,llama_schema= schema
)

chain = create_tagging_chain2(schema, llm)
inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
chain.run(inp)
{'sentiment': 'Positive', 'language': 'Spanish', 'stars': 4}

```
## The following have been tested with ChatGPTQ()

| Serial Number | model_name_or_path | model_basename |
| ------------- | ------------------ | -------------- |
| 1 | "TheBloke/LongChat-7B-GPTQ" | "longchat-7b-16k-GPTQ-4bit-128g.no-act.order" |
| 2 | "TheBloke/WizardLM-13B-V1-0-Uncensored-SuperHOT-8K-GPTQ" | "wizardlm-13b-v1.0-uncensored-superhot-8k-GPTQ-4bit-128g.no-act.order" |
| 3 | "TheBloke/Vicuna-33B-1-1-preview-SuperHOT-8K-GPTQ" | "vicuna-33b-1.3-preview-superhot-8k-GPTQ-4bit--1g.act.order" |


### Example 3
```python
`Chat_AutoModels()` can be used with any model by using `AutoTokenizer.from_pretrained(model_id)` and `AutoModelForCausalLM.from_pretrained(model_id)`.
from ChatOpenLLM import Chat_Llama, ChatGPTQ, Chat_AutoModels, create_tagging_chain2

llm = Chat_AutoModels(
    model_id="tiiuae/falcon-7b-instruct",
    device_map="auto",
    gen_kwargs=dict(temperature=0),
    #torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    load_in_4bit=True,
)

```
## Tested Models

The following models have been tested with ` Chat_AutoModels`:

| Serial Number | Model |
| ------------- | ------|
| 1             | tiiuae/falcon-7b-instruct |
| 2             | jondurbin/airoboros-13b |

 
