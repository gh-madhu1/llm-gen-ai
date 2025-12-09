"""
General Q&A Agent - Enhanced with Accuracy Improvements
Multi-stage pipeline: Query Analysis → Verification → Search → Synthesis
"""

import torch
from llm_gen_ai.utils import (
    load_pretrained_model,
    load_tokenizer_model,
    track_process_time,
    clear_device_cache
)
from llm_gen_ai.config import DEVICE, MODEL_PATH, GENERATION_CONFIG, MAX_MODEL_LENGTH

# Import new modules
from llm_gen_ai.modules.query_analyzer import QueryAnalyzer
from llm_gen_ai.modules.knowledge_verifier import KnowledgeVerifier
from llm_gen_ai.modules.parallel_search import ParallelSearchEngine
from llm_gen_ai.modules.answer_synthesizer import AnswerSynthesizer
from llm_gen_ai.modules.clarification_handler import ClarificationHandler
from llm_gen_ai.modules.answer_refiner import AnswerRefiner
from llm_gen_ai.modules.feedback_loop import FeedbackLoop


class EnhancedQAAgent:
    """Enhanced QA Agent with accuracy improvements."""

    def __init__(self, model, tokenizer, enable_verification=True):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
        
        # Initialize accuracy modules
        self.query_analyzer = QueryAnalyzer()
        self.knowledge_verifier = KnowledgeVerifier()
        self.search_engine = ParallelSearchEngine()
        self.answer_synthesizer = AnswerSynthesizer()
        self.clarification_handler = ClarificationHandler()
        self.answer_refiner = AnswerRefiner()
        self.feedback_loop = FeedbackLoop(max_iterations=3)
        
        self.enable_verification = enable_verification
    
    @track_process_time
    def generate_text(self, prompt, max_new_tokens=512, skip_cache_clear=True, stream=False):
        """
        Generate text response to a query.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            skip_cache_clear: Skip cache clearing for speed
            stream: Whether to stream tokens
        
        Returns:
            Generated response text (or yields tokens if streaming)
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_MODEL_LENGTH,
            padding=False
        ).to(DEVICE)

        if stream:
            # Streaming mode
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.get("attention_mask"),
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "streamer": streamer,
                **GENERATION_CONFIG
            }
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield new_text
            
            thread.join()
            
            if not skip_cache_clear:
                clear_device_cache()
            
            return generated_text
        else:
            # Non-streaming mode
            with torch.no_grad():
                generated = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    **GENERATION_CONFIG
                )

            full_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            response = full_text[len(prompt):].strip() if len(prompt) < len(full_text) else full_text.strip()
            
            if not skip_cache_clear:
                clear_device_cache()
            
            return response
    
    def answer_query(self, query, use_history=False, stream=False, show_analysis=True):
        """
        Answer query with full accuracy pipeline including clarification and refinement.
        
        Args:
            query: User's question
            use_history: Include conversation history
            stream: Stream the response
            show_analysis: Show query analysis and verification
            
        Returns:
            Generated answer (or generator if streaming)
        """
        # Step 1: Analyze Query
        print("\n📋 Analyzing query...")
        analysis = self.query_analyzer.analyze_query(query)
        
        if show_analysis:
            print(self.query_analyzer.format_analysis(analysis))
        
        # Step 2: Check for Clarification Needs (NEW)
        clarifications = None
        augmented_query = query
        
        if analysis.get('needs_clarification') or analysis.get('ambiguity_score', 0) > 0.5:
            if self.enable_verification and show_analysis:
                print(f"\n⚠️  Query ambiguity score: {analysis.get('ambiguity_score', 0):.2f}")
                
                # Generate and collect clarifying questions
                questions = self.clarification_handler.generate_clarifying_questions(query, analysis)
                
                if questions:
                    clarifications = self.clarification_handler.collect_user_responses(
                        questions, interactive=True
                    )
                    
                    if clarifications:
                        # Augment query with clarifications
                        augmented_query = self.clarification_handler.augment_query_with_responses(
                            query, clarifications
                        )
                        
                        if show_analysis:
                            print(self.clarification_handler.format_clarification_summary(
                                query, augmented_query, clarifications
                            ))
                        
                        # Re-analyze augmented query
                        print("\n🔄 Re-analyzing with clarifications...")
                        analysis = self.query_analyzer.analyze_query(augmented_query)
        
        # Step 3: Verify Knowledge
        assessment = self.knowledge_verifier.assess_knowledge(analysis)
        
        if show_analysis:
            print(self.knowledge_verifier.format_verification(assessment, analysis))
        
        # Step 4: Decide on Search (Enhanced)
        should_search = self.knowledge_verifier.should_search(assessment)
        search_results = None
        
        if should_search:
            # Confirm with user in interactive mode (if enabled)
            if self.enable_verification and show_analysis:
                confirmed = self.knowledge_verifier.get_user_confirmation()
                if not confirmed:
                    print("❌ Search cancelled by user.")
                    should_search = False
            
            if should_search:
                # Step 5: Parallel Search with Time Filtering (Enhanced)
                print("\n🌐 Searching web for latest information...")
                try:
                    search_results = self.search_engine.search_with_rewrites(
                        analysis['original'],
                        analysis['rewritten'],
                        max_results=3,
                        time_sensitive=analysis.get('time_sensitive', False)  # NEW
                    )
                    
                    if search_results.get('count', 0) > 0:
                        print(f"✅ Found {search_results['count']} relevant sources")
                    else:
                        print("⚠️  No search results found")
                        search_results = None
                except Exception as e:
                    print(f"⚠️  Search failed: {e}")
                    search_results = None
        
        # Step 6: Create Optimized Prompt
        prompt = self.answer_synthesizer.create_prompt(
            augmented_query,  # Use augmented query
            analysis,
            assessment,
            search_results,
            use_history,
            self.conversation_history if use_history else None
        )
        
        # Step 7: Generate Initial Answer
        print("\n⏱️ Generating answer...\n")
        print("✨ Answer:")
        
        if stream:
            # Streaming mode with refinement
            def stream_with_refinement():
                response = ""
                for token in self.generate_text(prompt, max_new_tokens=512, stream=True):
                    response += token
                    yield token
                
                # Step 8: Reflection & Refinement (NEW)
                print("\n\n🔍 Analyzing answer quality...")
                
                # Build context for refinement
                context = {
                    'query': query,
                    'analysis': analysis,
                    'assessment': assessment,
                    'search_results': search_results,
                    'clarifications': clarifications
                }
                
                # Assess initial answer
                original_assessment = self.answer_refiner.assess_answer_quality(
                    response, query, context
                )
                
                # Identify gaps
                gaps = self.answer_refiner.identify_gaps(response, query, context)
                
                # Refine if needed
                if gaps:
                    print(f"\n📝 Refining answer to address {len(gaps)} gap(s)...")
                    
                    # Create refinement function
                    def refine_func(refinement_prompt):
                        return self.generate_text(refinement_prompt, max_new_tokens=512, stream=False)
                    
                    refined_response = self.answer_refiner.refine_answer(
                        response, gaps, context, refine_func
                    )
                    
                    # Safety check: ensure refined_response is valid
                    if not refined_response or not isinstance(refined_response, str):
                        refined_response = response
                    
                    # Show refined answer
                    if refined_response != response:
                        yield "\n\n📊 Refined Answer:\n"
                        yield refined_response
                        response = refined_response
                    
                    # Reassess
                    refined_assessment = self.answer_refiner.assess_answer_quality(
                        refined_response, query, context
                    )
                else:
                    refined_assessment = original_assessment
                
                # Calculate final confidence
                confidence = self.answer_refiner.calculate_refined_confidence(
                    original_assessment, refined_assessment, context
                )
                
                # Show refinement summary
                summary = self.answer_refiner.format_refinement_summary(
                    gaps,
                    original_assessment['overall_score'],
                    refined_assessment['overall_score'],
                    confidence
                )
                yield summary
                
                # Synthesize final result with metadata
                result = self.answer_synthesizer.synthesize(
                    query, analysis, assessment, search_results, response
                )
                
                # Show metadata
                metadata = f"\n\n📊 Overall Confidence: {confidence:.2f}"
                yield metadata
                
                if result.get('sources'):
                    sources_text = self.search_engine.format_sources(result['sources'])
                    yield sources_text
                
                # Update conversation history
                self.conversation_history.append(f"User: {query}")
                self.conversation_history.append(f"Assistant: {response}")
            
            return stream_with_refinement()
        else:
            # Non-streaming mode with refinement
            response = self.generate_text(prompt, max_new_tokens=512)
            
            # Step 8: Reflection & Refinement (NEW)
            context = {
                'query': query,
                'analysis': analysis,
                'assessment': assessment,
                'search_results': search_results,
                'clarifications': clarifications
            }
            
            # Assess and refine
            original_assessment = self.answer_refiner.assess_answer_quality(
                response, query, context
            )
            gaps = self.answer_refiner.identify_gaps(response, query, context)
            
            if gaps and show_analysis:
                print(f"\n📝 Refining answer ({len(gaps)} gaps identified)...")
                
                def refine_func(refinement_prompt):
                    return self.generate_text(refinement_prompt, max_new_tokens=512, stream=False)
                
                response = self.answer_refiner.refine_answer(
                    response, gaps, context, refine_func
                )
                
                # Safety check: ensure response is valid string
                if not response or not isinstance(response, str):
                    # Fall back to original if refinement failed
                    response = self.generate_text(prompt, max_new_tokens=512)
            
            # Synthesize final answer  
            result = self.answer_synthesizer.synthesize(
                query, analysis, assessment, search_results, response
            )
            
            # Update conversation history
            self.conversation_history.append(f"User: {query}")
            self.conversation_history.append(f"Assistant: {response}")
            
            return result['answer']
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        print("✨ Conversation history cleared.")


# For backward compatibility - keep old class name working
class GeneralQAAgent(EnhancedQAAgent):
    """Backward compatible alias for EnhancedQAAgent."""
    pass


def interactive_mode(model, tokenizer):
    """
    Run the enhanced agent in interactive mode.
    
    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
    """
    agent = EnhancedQAAgent(model, tokenizer, enable_verification=True)
    
    print("\n" + "="*60)
    print("🤖 Enhanced Q&A Agent - Interactive Mode")
    print("="*60)
    print("\nFeatures:")
    print("  - Query analysis and rewriting")
    print("  - Knowledge verification")
    print("  - Parallel web search")
    print("  - Answer synthesis with citations")
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
            
            # Stream the answer with full analysis
            for token in agent.answer_query(query, use_history=True, stream=True, show_analysis=True):
                print(token, end="", flush=True)
            
            print("\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


@track_process_time
def answer_query(model, tokenizer, query, use_history=False):
    """
    Standalone function to answer a query (for compatibility).
    
    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
        query: User's question
        use_history: Whether to maintain conversation context
    
    Returns:
        Generated answer
    """
    agent = EnhancedQAAgent(model, tokenizer, enable_verification=False)
    return agent.answer_query(query, use_history, stream=False, show_analysis=True)


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Loading Enhanced Q&A Agent")
    print("=" * 60)
    print("\n📦 Loading model...")
    
    model = load_pretrained_model(MODEL_PATH)
    tokenizer = load_tokenizer_model(MODEL_PATH)
    
    # Run in interactive mode
    interactive_mode(model, tokenizer)
