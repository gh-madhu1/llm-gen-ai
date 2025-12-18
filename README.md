# llm-gen-ai

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

LLM-powered agents for question answering and document generation with enhanced accuracy through multi-stage pipelines, knowledge verification, and human-in-the-loop feedback.

## Features

### Enhanced QA Agent
- **Query Analysis & Rewriting** - Automatically analyzes and improves query clarity
- **Knowledge Verification** - Assesses internal knowledge and confidence levels
- **Parallel Web Search** - Concurrent search with time-sensitive filtering
- **Answer Synthesis** - Combines knowledge base and web search results with citations
- **Answer Refinement** - Automated quality assessment and iterative improvements
- **Clarification Handling** - Interactive clarification for ambiguous queries
- **Human Feedback Loop** - Iterative refinement based on user corrections

### White Paper Generation Agent
- **Multi-step Planning** - Structured outline generation
- **Research & Validation** - Novelty checking and web research
- **Context Memory** - Intelligent memory management to prevent loops
- **Document Export** - Professional Word document generation

### Model Quantization Support
- CPU-based inference with quantization
- Optimized for Apple Silicon (MPS) and CUDA GPUs
- Memory-efficient loading and caching

## Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/madhu1/llm-gen-ai.git
cd llm-gen-ai

# Install the package in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Using uv (recommended)

```bash
# Install dependencies using uv
uv pip install -e .
```

## Quick Start

### Enhanced QA Agent (Interactive Mode)

```bash
python examples/enhanced_qa_example.py
```

Or use the CLI command (after installation):

```bash
llm-qa-enhanced
```

### Single Query Mode

```bash
python examples/enhanced_qa_example.py "What is Model Context Protocol?"
```

### Basic QA Agent

```bash
python examples/basic_qa_example.py
```

### White Paper Generation

```bash
python examples/white_paper_example.py --idea "Your research topic here"
```

## Usage as a Library

```python
from llm_gen_ai import EnhancedQAAgent
from llm_gen_ai.core.model_loader import load_pretrained_model, load_tokenizer_model
from llm_gen_ai.config import MODEL_PATH

# Load model
model = load_pretrained_model(MODEL_PATH)
tokenizer = load_tokenizer_model(MODEL_PATH)

# Create agent
agent = EnhancedQAAgent(model, tokenizer, enable_verification=True)

# Ask questions
answer = agent.answer_query(
    "What is the latest development in AI?",
    use_history=False,
    stream=False,
    show_analysis=True
)

print(answer)
```

## Project Structure

```
llm-gen-ai/
├── src/llm_gen_ai/          # Main package
│   ├── agents/              # Agent implementations
│   ├── core/                # Core logic (loading, observability, memory, search)
│   ├── modules/             # Modular tasks (analysis, synthesis, etc.)
│   ├── config.py            # Global configuration
│   └── utils.py             # Shared utility functions
├── examples/                # Usage examples
├── experiments/             # Experimental model scripts
├── notebooks/               # Jupyter notebooks
├── docs/                    # Documentation
└── tests/                   # Test suite
```

## Configuration

Configure the agent in `src/llm_gen_ai/config.py`:

- `MODEL_PATH` - Path to your LLM model (Llama, Qwen, Gemma, etc.)
- `DEVICE` - Device for inference (cpu, cuda, mps)
- `GENERATION_CONFIG` - Temperature, top_p, and other generation parameters
- `MAX_MODEL_LENGTH` - Maximum context length
- `ENABLE_TIMING_LOGS` - Performance tracking

## Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get up and running quickly
- **[Agent Usage Guide](docs/AGENT_USAGE_GUIDE.md)** - Detailed agent usage
- **[QA Agent Documentation](docs/QA_AGENT.md)** - QA agent features and examples
- **[Optimization Guide](docs/OPTIMIZATION.md)** - Performance optimization tips

## Model Support

Tested with:
- Llama 3.2 (3B, 7B)
- Qwen 2.5 (3B)
- Gemma 3 (1B)

The framework supports any HuggingFace-compatible transformer model.

## Requirements

- Python 3.12+
- PyTorch 2.0+
- Transformers 4.33+
- 8GB+ RAM (16GB+ recommended)
- Optional: CUDA GPU or Apple Silicon for faster inference

## Model Quantization

For CPU-based machines, the system supports quantization to reduce memory usage:

```python
# In config.py
USE_QUANTIZATION = True
QUANTIZATION_BITS = 8  # or 4 for more aggressive quantization
```

See [`experiments/llama_7b.py`](experiments/llama_7b.py) for quantization examples.

## Examples

### Clarification Handling

```
💭 Question: What is MCP?

📋 Analyzing query...
⚠️  Query ambiguity score: 0.85

❓ Clarifying Questions:
1. MCP could refer to multiple things. Are you asking about:
   a) Master Control Program
   b) Model Context Protocol
   c) Microsoft Communication Platform
```

### Knowledge Verification

```
📋 Analyzing query...
✅ Knowledge Assessment:
   - Confidence: 0.65 (Medium)
   - Knowledge Level: Partial
   - Decision: Search recommended

🌐 Searching web for latest information...
✅ Found 3 relevant sources
```

### Answer Refinement

```
🔍 Analyzing answer quality...
📝 Refining answer to address 2 gap(s)...

📊 Refinement Summary:
   - Original Quality: 0.72
   - Refined Quality: 0.91
   - Confidence: 0.88
```

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=llm_gen_ai
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- Search powered by [DuckDuckGo](https://duckduckgo.com)
- Model support: Llama, Qwen, Gemma families

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{llm_gen_ai,
  title = {llm-gen-ai: LLM-powered Agents for QA and Document Generation},
  author = {Kanukula, Madhu},
  year = {2024},
  url = {https://github.com/madhu1/llm-gen-ai}
}
```
