# General Q&A Agent

A simple and efficient question-answering agent using the Llama 3.2 model for general queries.

## Overview

This agent is designed to answer general questions using a conversational AI approach. Unlike the white paper generation agent, this focuses on providing direct, helpful answers to user queries.

## Features

- **Single Query Mode**: Ask a single question and get an immediate answer
- **Interactive Mode**: Engage in a multi-turn conversation with conversation history
- **Optimized Performance**: Uses the same performance optimizations as the white paper agent
- **Memory Management**: Automatic cache clearing to prevent memory issues
- **Conversation History**: Maintains context across multiple questions in interactive mode

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### Method 1: Interactive Mode

Run the agent in interactive mode to have a conversation:

```bash
python run_qa_agent.py
```

This will start an interactive session where you can:
- Ask multiple questions
- Type `reset` to clear conversation history
- Type `exit` or `quit` to end the session

Example:
```
💭 Your question: What is machine learning?

🤔 Thinking...

✨ Answer:
Machine learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance on tasks without being explicitly programmed...

------------------------------------------------------------

💭 Your question: What are some common algorithms?

🤔 Thinking...

✨ Answer:
Common machine learning algorithms include linear regression, decision trees, random forests, support vector machines, neural networks...
```

### Method 2: Single Query Mode

Ask a single question from the command line:

```bash
python run_qa_agent.py "What is the capital of France?"
```

This will:
1. Load the model
2. Process your question
3. Print the answer
4. Exit

### Method 3: Use as a Module

Import and use the agent in your own Python code:

```python
from utils import load_pretrained_model, load_tokenizer_model
from general_qa_agent import GeneralQAAgent
from config import MODEL_PATH

# Load model
model = load_pretrained_model(MODEL_PATH)
tokenizer = load_tokenizer_model(MODEL_PATH)

# Create agent
agent = GeneralQAAgent(model, tokenizer)

# Ask questions
answer1 = agent.answer_query("What is Python?", use_history=False)
print(answer1)

# With conversation history
answer2 = agent.answer_query("What are its main features?", use_history=True)
print(answer2)

# Reset history
agent.reset_conversation()
```

## Configuration

The agent uses the same configuration as the white paper agent. You can modify settings in `config.py`:

- `MODEL_PATH`: Path to your Llama model
- `MAX_MODEL_LENGTH`: Maximum input length
- `GENERATION_CONFIG`: Generation parameters (temperature, top_p, etc.)

## Performance Tips

1. **First Load**: The first query will be slower as the model loads into memory
2. **Subsequent Queries**: Much faster once the model is loaded
3. **Memory**: The agent automatically clears cache to prevent memory issues
4. **Conversation History**: Limited to last 4 messages (2 Q&A pairs) to manage context length

## Comparison with White Paper Agent

| Feature | White Paper Agent | General Q&A Agent |
|---------|------------------|-------------------|
| Purpose | Create comprehensive documents | Answer general questions |
| Complexity | High (multi-step reasoning) | Low (direct responses) |
| Output | Structured document | Text answer |
| Tools | PLAN, SEARCH, WRITE, etc. | Simple text generation |
| Use Case | Research & documentation | Quick Q&A |

## Examples

### Example 1: Technical Question
```
Question: Explain the difference between a list and a tuple in Python
Answer: Lists are mutable sequences in Python, meaning you can modify them after creation...
```

### Example 2: General Knowledge
```
Question: What are the benefits of renewable energy?
Answer: Renewable energy sources like solar, wind, and hydroelectric power offer several key benefits...
```

### Example 3: Conversational
```
Question: Tell me about Mars
Answer: Mars is the fourth planet from the Sun, often called the Red Planet...

Question: How far is it from Earth?
Answer: The distance between Earth and Mars varies significantly...
```

## Troubleshooting

### Out of Memory Error
- The agent automatically clears cache, but if you still encounter issues:
  - Reduce `max_new_tokens` parameter
  - Use single query mode instead of interactive mode
  - Restart the agent periodically

### Slow Responses
- First query is always slower (model loading)
- Ensure you're using GPU if available (check `DEVICE` in config)
- Consider using a smaller model if needed

### Poor Quality Answers
- Try rephrasing your question
- Be more specific in your query
- Use conversation history for follow-up questions

## Files

- `general_qa_agent.py`: Core Q&A agent implementation
- `run_qa_agent.py`: Command-line runner script
- `config.py`: Configuration settings (shared with white paper agent)
- `utils.py`: Utility functions (shared with white paper agent)

## License

Apache 2.0 License (same as the parent project)
