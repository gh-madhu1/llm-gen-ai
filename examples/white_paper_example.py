#!/usr/bin/env python3
"""
Main entry point for the Optimized White Paper Generation Agent.
Provides a clean interface to run the agent with command-line arguments.
"""
import argparse
import sys
from llm_gen_ai.core.model_loader import load_pretrained_model, load_tokenizer_model
from llm_gen_ai.utils import track_process_time
from llm_gen_ai.agents.white_paper_agent import generate
from llm_gen_ai.config import MODEL_PATH


@track_process_time
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive white papers using AI"
    )
    parser.add_argument(
        "--idea",
        type=str,
        help="The white paper idea/topic (if not provided, uses default example)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Model path (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated files (default: current directory)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in benchmark mode to measure performance"
    )
    
    args = parser.parse_args()

    # Load model and tokenizer (cached after first call)
    print("="*60)
    print("🚀 Optimized White Paper Generation Agent")
    print("="*60)
    print(f"\n📦 Loading model: {args.model}")
    
    try:
        model = load_pretrained_model(args.model)
        tokenizer = load_tokenizer_model(args.model)
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("Please ensure the model is available and you have internet connection.")
        sys.exit(1)

    # Use provided idea or default example
    if args.idea:
        idea = args.idea
    else:
        print("\n💡 Using default example idea...")
        idea = """
        Explore Adaptive Agent Selection for Multi-Agent LLM Systems, Focused on:
        - The supervisor agent will choose the best agent based on the feedback given to each response when we routed the request to the different agents for the same query. Based on self agent as a judge rate and human feedback the models/agents which performing well should be used to respond to the queries.
        - This helps the best accurate results served to the user. Helps the trustworthy of the AI Usage in realtime use cases.
        - You can write a paper explaining how we can implement and achieve the solution.
        - Highlight any potential risks and issues.
        - Write how it can help optimize the costs, and highlight if any other optimization thoughts.
        """

    print(f"\n📝 Topic: {idea[:150]}...")
    print("\n" + "="*60 + "\n")

    # Generate white paper
    try:
        result = generate(model, tokenizer, idea, args.output_dir)
        print(f"\n{'='*60}")
        print(f"✅ Result: {result}")
        print(f"{'='*60}")
        
        if args.benchmark:
            print("\n📊 Benchmark mode: Check timing logs above for performance metrics")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
