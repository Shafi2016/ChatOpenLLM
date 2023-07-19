from setuptools import setup, find_packages

setup(
    name="ChatOpenLLM",
    version="0.1",
    packages=find_packages(),
    description="Open-source Language Model based on Langchain, for creating ChatGPT-like applications.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Shafiullah Qureshi, Saad Hassan Khan",
    author_email="qureshi.shafiullah@gmail.com",
    url="https://github.com/Shafi2016/ChatOpenLLM",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "torch",
        "transformers",
        "langchain",
        "sentence_transformers",
        "accelerate",
        "sentencepiece",
        "auto_gptq",
        "safetensors",
        "kor",
        "bitsandbytes",
        "Xformers",
    ],
)
