import sys
import os
from typing import Annotated, Dict, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
# from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, END
from llm_gen_ai.core.llm import get_langchain_llm

# Ensure you have OPENAI_API_KEY environment variable set
# Ensure you have LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST environment variables set

class AgentState(TypedDict):
    messages: list[BaseMessage]

class LangGraphQAAgent:
    def __init__(self):
        # Initialize Langfuse Callback
        # Initialize Langfuse Callback
        # If running locally with Docker, the default host might work. 
        # If not, set LANGFUSE_HOST environment variable.
        # self.langfuse_handler = CallbackHandler()
        
        self.llm = get_langchain_llm()
        
        # Build the graph
        self.app = self._build_graph()

    def _call_model(self, state: AgentState):
        messages = state['messages']
        # print(f"DEBUG: Messages passed to LLM: {messages}")
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", self._call_model)
        
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)
        
        return workflow.compile()

    def answer_question(self, question: str, history: list[BaseMessage] = None):
        if history is None:
            history = []
            
        inputs = {"messages": history + [HumanMessage(content=question)]}
        
        # Pass callbacks in config
        # config = {"callbacks": [self.langfuse_handler]}
        config = {}
        
        # Invoke the graph
        result = self.app.invoke(inputs, config=config)
        
        # Return the last message content
        return result["messages"][-1].content

def main():
    agent = LangGraphQAAgent()
    
    # Single-shot mode
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"Question: {question}")
        answer = agent.answer_question(question)
        print("\nAnswer:")
        print(answer)
        return

    # Interactive mode
    print("\n🕸️ LangGraph Agent Interactive Mode")
    print("Type 'exit' or 'quit' to stop.\n")
    
    history = []
    
    while True:
        try:
            question = input("\n👉 Enter your question: ")
            if question.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not question.strip():
                continue
            
            # Get answer with history
            answer = agent.answer_question(question, history)
            
            # Print Answer
            print("\nAnswer:")
            print(answer)
            
            # Update history manually (since we want to maintain state across calls if not managed by graph memory checkpointer)
            # The simple graph returns all messages, but for client-side history we append pairs.
            # However, the current graph is stateless between invokes unless checkpointing is used.
            # To simulate conversation, we pass history in.
            history.append(HumanMessage(content=question))
            # Ideally the agent response is also a BaseMessage. 
            # But here we just get string. Let's create AIMessage to store in history.
            from langchain_core.messages import AIMessage
            history.append(AIMessage(content=answer))
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
