"""
Enhanced QA Agent with Human Feedback Loop - Example Usage

This script demonstrates the human feedback loop feature.
"""

from llm_gen_ai.utils import load_pretrained_model, load_tokenizer_model
from llm_gen_ai.agents.enhanced_qa_agent import EnhancedQAAgent
from llm_gen_ai.config import MODEL_PATH


def test_feedback_loop():
    """Test the feedback loop functionality."""
    
    print("="*60)
    print("Testing Human Feedback Loop")
    print("="*60)
    
    # Load model
    print("\n📦 Loading model...")
    model = load_pretrained_model(MODEL_PATH)
    tokenizer = load_tokenizer_model(MODEL_PATH)
    
    # Create agent with feedback enabled
    agent = EnhancedQAAgent(model, tokenizer, enable_verification=True)
    
    # Example query that might need correction
    query = "What is Model Context Protocol?"
    
    print(f"\n💭 Question: {query}")
    print("\nThe agent will:")
    print("  1. Generate an initial answer")
    print("  2. Present it as a DRAFT")
    print("  3. Ask for your feedback")
    print("  4. Refine based on your corrections")
    print("  5. Repeat until you confirm it's correct")
    print("\n" + "-"*60)
    
    # This will trigger the full pipeline with feedback
    # The feedback loop is currently in the streaming path
    # For non-streaming with feedback, we need to update answer_query
    
    print("\nNote: Full feedback integration in progress.")
    print("Current version supports:")
    print("  ✓ Clarifying questions for ambiguous queries")
    print("  ✓ Web search with time filtering")
    print("  ✓ Automated answer refinement")
    print("  ⏳ Human feedback loop (in development)")


if __name__ == "__main__":
    test_feedback_loop()
