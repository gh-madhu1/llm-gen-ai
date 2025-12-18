"""
Core module for LLM wrappers and LangChain integration.
"""
from typing import Any, List, Optional
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from llm_gen_ai.core.model_loader import load_pretrained_model, load_tokenizer_model

class LocalHuggingFaceChat(BaseChatModel):
    """
    A local Chat Model wrapper for HuggingFace Pipeline to satisfy CrewAI/LangChain Chat Interface.
    """
    llm: Any  # HuggingFacePipeline
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Use tokenizer's chat template
        # Access the underlying pipeline's tokenizer
        tokenizer = self.llm.pipeline.tokenizer
        
        # Convert LangChain messages to HF format
        hf_messages = []
        for m in messages:
            role = "user"
            if m.type == "ai":
                role = "assistant"
            elif m.type == "system":
                role = "system"
            elif m.type == "human":
                role = "user"
            
            hf_messages.append({"role": role, "content": m.content})
            
        # Apply chat template
        # We use tokenize=False to get the string prompt
        try:
            prompt = tokenizer.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            # Fallback for simple prompt construction if template fails
            print(f"Warning: apply_chat_template failed: {e}")
            prompt = ""
            for m in messages:
                prompt += f"{m.type}: {m.content}\n"
            prompt += "assistant: "

        # Generate (invoke returns string)
        response_text = self.llm.invoke(prompt, stop=stop)
        
        # Clean response if it includes the prompt (pipeline usually returns generated text only if specified, 
        # but invoke might return full text. We need to check.)
        # HuggingFacePipeline's invoke usually returns only the generated text if `text_generation` pipeline is used with `return_full_text=False`.
        # But we initialized it default.
        # Let's assume it returns what we need or clean it.
        # If response starts with prompt, strip it.
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):]
            
        message = AIMessage(content=response_text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "local_huggingface_chat"


def get_langchain_llm():
    """
    Returns a LangChain compatible LLM using the locally loaded model.
    """
    model = load_pretrained_model()
    tokenizer = load_tokenizer_model()
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False, # Critical for chat mode
        streamer=streamer, # Enable streaming to stdout
        # device_map="auto" # handled by pipeline/model loading
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Wrap in our custom Chat class
    chat_model = LocalHuggingFaceChat(llm=llm)
    
    return chat_model
