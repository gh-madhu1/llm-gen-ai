"""
General Q&A Agent - Simple query answering system with web search
This agent responds to general queries using the LLM model and can search the web for current information.
"""

import torch
from llm_gen_ai.utils import (
    load_pretrained_model,
    load_tokenizer_model,
    track_process_time,
    clear_device_cache
)
from llm_gen_ai.config import DEVICE, MODEL_PATH, GENERATION_CONFIG, MAX_MODEL_LENGTH
from llm_gen_ai.core.search_engine import SearchEngine


class GeneralQAAgent:
    """Agent for answering general questions with web search capability."""

    def __init__(self, model, tokenizer, enable_search=True):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
        self.search_engine = SearchEngine() if enable_search else None
        self.enable_search = enable_search

    @track_process_time
    def generate_text(self, prompt, max_new_tokens=512, skip_cache_clear=True, stream=False):
        """
        Generate text response to a query.
        OPTIMIZED: Improved tokenization and cache management.
        
        Args:
            prompt: Input question/query
            max_new_tokens: Maximum tokens to generate
            skip_cache_clear: Skip cache clearing for speed (default: True)
            stream: Whether to stream tokens as they're generated (default: False)
        
        Returns:
            Generated response text (or yields tokens if streaming)
        """
        # Tokenize with optimized settings
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_MODEL_LENGTH,
            padding=False  # No padding needed for single input
        ).to(DEVICE)

        if stream:
            # Use streaming for real-time token generation
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Generation parameters
            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.get("attention_mask"),
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "streamer": streamer,
                **GENERATION_CONFIG
            }
            
            # Run generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Yield tokens as they arrive
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield new_text
            
            thread.join()
            
            # Clear cache only when needed
            if not skip_cache_clear:
                clear_device_cache()
            
            return generated_text
        else:
            # Non-streaming generation
            with torch.no_grad():
                generated = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    **GENERATION_CONFIG
                )

            # Decode response
            full_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            # Extract only the new generated text
            response = full_text[len(prompt):].strip()
            
            # Clear cache only when needed (adaptive strategy)
            if not skip_cache_clear:
                clear_device_cache()
            
            return response

    def _needs_search(self, query):
        """
        Determine if a query needs web search based on keywords.
        
        Args:
            query: User's question
            
        Returns:
            bool: True if search is needed
        """
        query_lower = query.lower()
        
        # Keywords that indicate need for current information
        current_keywords = [
            'latest', 'recent', 'current', 'today', 'now', 'this year',
            'news', 'update', '2024', '2025', 'what happened',
            'price', 'stock', 'weather', 'score', 'result'
        ]
        
        # Check if query contains current event indicators
        return any(keyword in query_lower for keyword in current_keywords)
    
    def _search_web(self, query, max_results=3):
        """
        Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Formatted search results string
        """
        if not self.enable_search or not self.search_engine:
            return None
        
        try:
            results = self.search_engine.search(query, max_results=max_results)
            return results
        except Exception as e:
            print(f"⚠️  Search failed: {e}")
            return None

    def answer_query(self, query, use_history=False, stream=False, auto_search=True):
        """
        Answer a general query with optional web search.
        
        Args:
            query: User's question
            use_history: Whether to include conversation history
            stream: Whether to stream the response token-by-token
            auto_search: Whether to automatically search web when needed
        
        Returns:
            Generated answer (or generator if streaming)
        """
        # Determine if search is needed
        search_results = None
        if auto_search and self.enable_search and self._needs_search(query):
            print("🌐 Searching the web for latest information...")
            search_results = self._search_web(query)
        
        # Build prompt with search context if available
        if search_results:
            # Include search results in the prompt
            if use_history and self.conversation_history:
                context = "\n".join(self.conversation_history[-4:])
                prompt = f"""{context}

Web Search Results:
{search_results}

Using the above search results and your knowledge, answer the following question:

User: {query}
Assistant:"""
            else:
                prompt = f"""You are a helpful AI assistant. Use the following web search results to provide an accurate, up-to-date answer.

Web Search Results:
{search_results}

Question: {query}

Provide a comprehensive answer based on the search results above. If the question is ambiguous or unclear, ask clarifying questions before providing a full answer.
Assistant:"""
        else:
            # No search - use regular prompt with clarification instructions
            if use_history and self.conversation_history:
                # Include previous conversation context
                context = "\n".join(self.conversation_history[-4:])  # Last 2 Q&A pairs
                prompt = f"""{context}

User: {query}
Assistant:"""
            else:
                # Simple query without history - encourage clarification
                prompt = f"""You are a helpful AI assistant. Your goal is to provide clear, accurate answers.

IMPORTANT GUIDELINES:
1. If the question is ambiguous, vague, or could have multiple interpretations, ask 2-3 specific clarifying questions before answering.
2. If the question uses unclear abbreviations or acronyms that could mean different things, ask which specific meaning the user intends.
3. If the question lacks necessary context, ask for relevant details.
4. Only provide a full answer when you're confident you understand what the user is asking.

Examples of when to ask for clarification:
- "What is MCP?" → Could be many things. Ask: "MCP could refer to several things like Master Control Program, Model Context Protocol, or Microsoft Communication Platform. Which one are you asking about?"
- "Tell me about Python" → Ask: "Are you asking about the Python programming language, the snake species, or something else?"
- "How does it work?" → Ask: "What specific system or concept are you asking about?"

User: {query}
Assistant:"""

        if stream:
            # Streaming mode - return generator
            def stream_with_history():
                response = ""
                for token in self.generate_text(prompt, max_new_tokens=512, stream=True):
                    response += token
                    yield token
                
                # Update conversation history after streaming completes
                self.conversation_history.append(f"User: {query}")
                self.conversation_history.append(f"Assistant: {response}")
            
            return stream_with_history()
        else:
            # Non-streaming mode
            response = self.generate_text(prompt, max_new_tokens=512)
            
            # Update conversation history
            self.conversation_history.append(f"User: {query}")
            self.conversation_history.append(f"Assistant: {response}")
            
            return response

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        print("✨ Conversation history cleared.")


@track_process_time
def answer_query(model, tokenizer, query, use_history=False):
    """
    Generate an answer to a general query.
    
    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
        query: User's question
        use_history: Whether to maintain conversation context
    
    Returns:
        Generated answer
    """
    agent = GeneralQAAgent(model, tokenizer)
    return agent.answer_query(query, use_history)


def interactive_mode(model, tokenizer):
    """
    Run the agent in interactive Q&A mode with streaming support.
    
    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
    """
    agent = GeneralQAAgent(model, tokenizer)
    
    print("\n" + "="*60)
    print("🤖 General Q&A Agent - Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  - Type your question to get an answer")
    print("  - Type 'reset' to clear conversation history")
    print("  - Type 'exit' or 'quit' to end the session")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("\n💭 Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'reset':
                agent.reset_conversation()
                continue
            
            print("\n🤔 Generating answer...\n")
            print("✨ Answer:")
            
            # Stream the answer token-by-token
            for token in agent.answer_query(query, use_history=True, stream=True):
                print(token, end="", flush=True)
            
            print("\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            continue


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Loading General Q&A Agent")
    print("=" * 60)
    print("\n📦 Loading model...")
    
    model = load_pretrained_model(MODEL_PATH)
    tokenizer = load_tokenizer_model(MODEL_PATH)
    
    # Run in interactive mode
    interactive_mode(model, tokenizer)
