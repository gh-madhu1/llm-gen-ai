
import os
import sys
import json
from llm_gen_ai.agents.enhanced_qa_agent import EnhancedQAAgent
from llm_gen_ai.core.model_loader import load_pretrained_model, load_tokenizer_model
from llm_gen_ai.config import MODEL_PATH

def verify_parallel_flow():
    print("🚀 Starting Parallel Flow & Rewriting Verification")
    
    log_file = "agent_activity.jsonl"
    if os.path.exists(log_file):
        os.remove(log_file)
        
    print("📦 Loading model...")
    try:
        model = load_pretrained_model(MODEL_PATH)
        tokenizer = load_tokenizer_model(MODEL_PATH)
        agent = EnhancedQAAgent(model, tokenizer, enable_verification=True)
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    # Test Query likely to trigger search and have internal knowledge
    # "Who is the CEO of Google?" (Internal knowledge usually knows Sundar Pichai, Search confirms)
    query = "Who is the CEO of Google and what AI model did they recently release?"
    
    print(f"\n🧪 Processing Query: '{query}'")
    answer = agent.answer_query(query, show_analysis=False)
    
    print("\n📝 Generated Answer Check:")
    if answer:
        print(f"Answer length: {len(answer)} chars")
        print(f"Preview: {answer[:100]}...")
    else:
        print("❌ No answer generated")
    
    # Check Logs for specific markers
    print("\n🔍 Verifying Operation Logs...")
    found_rewrite = False
    found_parallel_metrics = False
    
    search_latency = 0
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                log = json.loads(line)
                
                # Check for rewrite log
                if "Query rewritten by LLM" in log.get('message', ''):
                    found_rewrite = True
                    print(f"  - ✅ Rewriting occurred: {log.get('rewritten')}")
                
                # Check for search latency (implies parallel block ran)
                if log.get('metric_name') == 'web_search_latency':
                    found_parallel_metrics = True
                    search_latency = log.get('value')
                    print(f"  - ✅ Search executed in parallel block (Latency: {search_latency}s)")

    if found_rewrite and found_parallel_metrics:
        print("\n✅ VERIFICATION SUCCESSFUL: Rewriting and Parallel Search active.")
    else:
        print("\n❌ VERIFICATION FAILED: Missing steps.")
        print(f"  Rewriting Found: {found_rewrite}")
        print(f"  Parallel Search Found: {found_parallel_metrics}")

if __name__ == "__main__":
    verify_parallel_flow()
