# LLM Agent Usage Guide

This repository now contains **two different agent modes**:

## 🤖 Option 1: General Q&A Agent (NEW)

**Purpose**: Answer general questions and queries

**Use Cases**:
- Quick questions and answers
- General knowledge queries
- Technical explanations
- Conversational AI interactions

**How to Run**:

### Single Question
```bash
python run_qa_agent.py "Your question here"
```

### Interactive Mode
```bash
python run_qa_agent.py
```
Then type your questions interactively. Commands:
- Type your question to get an answer
- Type `reset` to clear conversation history
- Type `exit` or `quit` to end

**Example**:
```bash
python run_qa_agent.py "What is artificial intelligence?"
```

**Documentation**: See [README_QA_AGENT.md](README_QA_AGENT.md)

**Files**:
- `general_qa_agent.py` - Core Q&A agent
- `run_qa_agent.py` - Command-line runner

---

## 📄 Option 2: White Paper Generation Agent

**Purpose**: Generate comprehensive, publication-ready white papers

**Use Cases**:
- Research papers
- Technical documentation
- Comprehensive reports
- Multi-section documents with references

**How to Run**:
```bash
python Llama-3.2-3B-Instruct_v2.py
```
or
```bash
python main.py
```

**Documentation**: See [README_OPTIMIZED.md](README_OPTIMIZED.md)

**Files**:
- `Llama-3.2-3B-Instruct_v2.py` - Optimized white paper agent
- `agent_core.py` - Core agent logic
- `main.py` - Alternative entry point

---

## Comparison

| Feature | General Q&A Agent | White Paper Agent |
|---------|------------------|-------------------|
| **Speed** | Fast (30-60s per query) | Slow (several minutes) |
| **Output** | Text answer | Structured document (.docx) |
| **Complexity** | Simple | Complex (multi-step) |
| **Tools** | None | PLAN, SEARCH, WRITE, FINISH |
| **Best For** | Quick answers | Research & documentation |
| **Interactive** | ✅ Yes | ❌ No |
| **Conversation History** | ✅ Yes | ❌ No |

---

## Quick Start

### For General Questions
```bash
# Install dependencies
pip install -r requirements.txt

# Ask a question
python run_qa_agent.py "Explain machine learning"
```

### For White Paper Generation
```bash
# Install dependencies
pip install -r requirements.txt

# Generate a white paper (edit the idea in the file first)
python Llama-3.2-3B-Instruct_v2.py
```

---

## Configuration

Both agents share the same configuration file (`config.py`) with settings for:
- Model path
- Device (cpu/cuda/mps)
- Generation parameters
- Memory management
- Performance tuning

---

## Model Requirements

Both agents use the **Llama 3.2 3B Instruct** model:
- Model: `meta-llama/Llama-3.2-3B-Instruct`
- Size: ~3 billion parameters
- Recommended: 8GB+ RAM/VRAM
- GPU recommended but not required

---

## Performance

### General Q&A Agent
- First query: ~30-60 seconds (model loading)
- Subsequent queries: ~10-30 seconds
- Memory: ~4-6GB

### White Paper Agent
- Total time: 5-15 minutes
- Sections: 8-12 sections
- Memory: ~4-8GB
- Output: Professional .docx document

---

## Choose Your Mode

**Need a quick answer?** → Use the **General Q&A Agent**

**Need a comprehensive document?** → Use the **White Paper Agent**

Both modes use the same underlying model and optimization techniques!
