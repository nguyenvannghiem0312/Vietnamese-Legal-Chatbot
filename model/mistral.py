from client.llm_client import LlmClientType
from model.model import Model


class MistralSettings(Model):
    url = "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_K_M.gguf"
    file_name = "mistral-7b-openorca.Q4_K_M.gguf"
    clients = [LlmClientType.CTRANSFORMERS]
    type = "mistral"
    """
    Config:
    - top_k="The top-k value to use for sampling."
    - top_p="The top-p value to use for sampling."
    - temperature="The temperature to use for sampling."
    - repetition_penalty="The repetition penalty to use for sampling."
    - last_n_tokens="The number of last tokens to use for repetition penalty."
    - seed="The seed value to use for sampling tokens."
    - max_new_tokens="The maximum number of new tokens to generate."
    - stop="A list of sequences to stop generation when encountered."
    - stream="Whether to stream the generated text."
    - reset="Whether to reset the model state before generating text."
    - batch_size="The batch size to use for evaluating tokens in a single prompt."
    - threads="The number of threads to use for evaluating tokens."
    - context_length="The maximum context length to use."
    - gpu_layers="The number of layers to run on GPU."
        - Set gpu_layers to the number of layers to offload to GPU.
        - Set to 0 if no GPU acceleration is available on your system.
    """

    config = {
        "top_k": 40,
        "top_p": 0.95,
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "last_n_tokens": 64,
        "seed": -1,
        "batch_size": 8,
        "threads": -1,
        "max_new_tokens": 1024,
        "stop": None,
        "stream": False,
        "reset": True,
        "context_length": 2048,
        "gpu_layers": 50,
        "mmap": True,
        "mlock": False,
    }
    system_template = "You are a helpful, respectful and honest assistant."
    qa_prompt_template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    ctx_prompt_template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question below:
{question}<|im_end|>
<|im_start|>assistant
"""
    refined_ctx_prompt_template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
The original query is as follows: {question}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer
(only if needed) with some more context below.
---------------------
{context}
---------------------
Given the new context, refine the original answer to better answer the query.
If the context isn't useful, return the original answer.
Refined Answer:<|im_end|>
<|im_start|>assistant
"""
    refined_question_conversation_awareness_prompt_template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
\nChat History:
---------------------
{chat_history}
---------------------
Follow Up Question: {question}
Given the above conversation and a follow up question, rephrase the follow up question to be a standalone question.
Standalone question:<|im_end|>
<|im_start|>assistant
"""

    refined_answer_conversation_awareness_prompt_template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
\nChat History:
---------------------
{chat_history}
---------------------
Considering the context provided in the Chat History, answer the question below with conversation awareness:
{question}<|im_end|>
<|im_start|>assistant
"""