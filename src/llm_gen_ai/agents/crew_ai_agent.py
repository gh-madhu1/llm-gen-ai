import os
import sys
from crewai import Agent, Task, Crew, Process
from crewai import Agent, Task, Crew, Process
# from langfuse.langchain import CallbackHandler
from llm_gen_ai.core.llm import get_langchain_llm

# Ensure you have OPENAI_API_KEY environment variable set
# Ensure you have LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST environment variables set

# Ensure you have LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST environment variables set

# Workaround for CrewAI strict validation when using local models
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "NA"

class CrewAIQAAgent:
    def __init__(self):
        # Initialize Langfuse Callback
        # If running locally with Docker, the default host might work. 
        # If not, set LANGFUSE_HOST environment variable.
        # self.langfuse_handler = CallbackHandler()
        
        # Use local LLM
        self.llm = get_langchain_llm()
        # Note: Depending on how CrewAI handles callbacks for custom LLMs, 
        # we might need to attach the handler differently or it might propagate.
        # But HuggingFacePipeline wrapper generally accepts callbacks in invoke/generate, 
        # but here we pass the llm object.
        # Ideally, we attach the callback handler to the LLM if supported.
        # Since we're using a pipeline, we might need to rely on the agent-level callbacks if supported by CrewAI,
        # or rely on the global Langchain callback manager if configured.
        # For now, we rely on Langfuse autoinstrumentation if possible or pass it where allowed.
        # However, to be explicit with Langfuse handler on the LLM:
         # self.llm.callbacks = [self.langfuse_handler]
        
        # Define the Agent once
        self.researcher = Agent(
            role='Research Analyst',
            goal='Provide accurate and concise answers to user questions.',
            backstory='You are an expert analyst who can explain complex topics simply.',
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        print(f"DEBUG: Using LLM type: {type(self.llm)}")

    def answer_question(self, question: str, context: str = ""):
        # Define the Task
        description = f"Answer the following question: {question}"
        if context:
            description = f"Context from previous conversation:\n{context}\n\nCurrent Question: {question}"
            
        task = Task(
            description=description,
            agent=self.researcher,
            expected_output="A concise and accurate answer to the question."
        )

        # Define the Crew
        crew = Crew(
            agents=[self.researcher],
            tasks=[task],
            verbose=True, 
            process=Process.sequential
        )

        # Kickoff the process
        result = crew.kickoff()
        return result

def main():
    agent = CrewAIQAAgent()
    
    # Check for command line arguments for single-shot mode
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"Question: {question}")
        answer = agent.answer_question(question)
        print("\nAnswer:")
        print(answer)
        return

    # Interactive mode
    print("\n🤖 CrewAI Agent Interactive Mode")
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
            
            # Build context string
            context_str = "\n".join(history)
            
            answer = agent.answer_question(question, context=context_str)
            print("\nAnswer:")
            print(answer)
            
            # Update history
            history.append(f"User: {question}")
            history.append(f"AI: {answer}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
