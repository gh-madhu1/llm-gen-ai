import os
# Set dummy keys to prevent implicit OpenAI usage if any
os.environ["OPENAI_API_KEY"] = "NA"
# Updated import for the new structure
from llm_gen_ai.core.llm import get_langchain_llm

def test():
    print("Loading LLM...")
    llm = get_langchain_llm()
    print("LLM Loaded. Invoking...")
    try:
        response = llm.invoke("Hello, simple test.")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
