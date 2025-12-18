#!/usr/bin/env python3
"""
Enhanced QA Agent Runner
Uses improved accuracy pipeline with query analysis, verification, and synthesis.
"""

from llm_gen_ai.core.model_loader import load_pretrained_model, load_tokenizer_model
from llm_gen_ai.agents.enhanced_qa_agent import interactive_mode, EnhancedQAAgent
from llm_gen_ai.config import MODEL_PATH
import sys


def main():
    """Main entry point."""
    print("="*60)
    print("🚀 Enhanced General Q&A Agent")
    print("="*60)
    print("\n📦 Loading model...")
    
    try:
        model = load_pretrained_model(MODEL_PATH)
        tokenizer = load_tokenizer_model(MODEL_PATH)
        
        # Check if a question was provided as command-line argument
        if len(sys.argv) > 1:
            # Single question mode with analysis
            query = " ".join(sys.argv[1:])
            print(f"\n💭 Question: {query}")
            
            # Create agent (no interactive verification for CLI)
            agent = EnhancedQAAgent(model, tokenizer, enable_verification=False)
            
            # Stream the answer with analysis
            for token in agent.answer_query(query, use_history=False, stream=True, show_analysis=True):
                print(token, end="", flush=True)
            
            print("\n")
            print("="*60)
        else:
            # Interactive mode with full features
            interactive_mode(model, tokenizer)
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
