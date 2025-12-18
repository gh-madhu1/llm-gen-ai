
import os
import sys
import json
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from llm_gen_ai.agents.enhanced_qa_agent import EnhancedQAAgent
from llm_gen_ai.core.model_loader import load_pretrained_model, load_tokenizer_model
from llm_gen_ai.config import MODEL_PATH

def test_observability():
    print("🚀 Starting Observability Verification")
    
    # 1. Setup
    log_file = "agent_activity.jsonl"
    if os.path.exists(log_file):
        os.remove(log_file)
        
    print("📦 Loading model (mocking for speed if needed, but using real for now)...")
    # For a real test we'd load the model, but to verify logic we can potentially mock
    # But let's assume the environment is set up for it.
    try:
        model = load_pretrained_model(MODEL_PATH)
        tokenizer = load_tokenizer_model(MODEL_PATH)
        agent = EnhancedQAAgent(model, tokenizer, enable_verification=True)
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    # 2. Test Case 1: Time-sensitive Query (Should Search automatically)
    print("\n🧪 Test Case 1: Time-sensitive Query (Expect Search)")
    query_search = "What are the latest AI models released in 2024?"
    
    try:
        # We use answer_query which now shouldn't prompt for input
        response = agent.answer_query(query_search, show_analysis=False)
        print("✅ Query processing complete (no user input requested)")
    except OSError as e:
        print("❌ FAILED: Agent likely requested user input (OSError usually means stdin read failure in non-interactive)")
        return

    # 3. Test Case 2: General Knowledge Query (Expect No Search)
    print("\n🧪 Test Case 2: General Knowledge Query (Expect No Search)")
    query_know = "What is a neural network?"
    agent.answer_query(query_know, show_analysis=False)

    # 4. Verify Logs
    print("\n🔍 Verifying Logs...")
    if not os.path.exists(log_file):
        print("❌ FAILED: Log file not found")
        return

    found_search_decision = False
    found_skip_decision = False
    found_latency_metric = False
    
    with open(log_file, 'r') as f:
        for line in f:
            log = json.loads(line)
            # print(log) # Debug
            
            if log.get('event_type') == 'agent_decision':
                if log.get('decision') == 'search':
                    found_search_decision = True
                    print("  - Found Search decision log ✅")
                elif log.get('decision') == 'skip_search':
                    found_skip_decision = True
                    print("  - Found Skip Search decision log ✅")
            
            if log.get('event_type') == 'metric' and 'latency' in log.get('metric_name', ''):
                found_latency_metric = True

    if found_search_decision and found_skip_decision and found_latency_metric:
        print("\n✅ VERIFICATION SUCCESSFUL: Observability and Logic working as expected.")
    else:
        print("\n❌ VERIFICATION FAILED: Missing expected logs.")
        print(f"  Search Decision Found: {found_search_decision}")
        print(f"  Skip Decision Found: {found_skip_decision}")
        print(f"  Latency Metric Found: {found_latency_metric}")

if __name__ == "__main__":
    test_observability()
