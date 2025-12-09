# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete repository reorganization following Python best practices
- Proper package structure with `src/llm_gen_ai/`
- Comprehensive project documentation
- Enhanced `pyproject.toml` with build configuration and entry points
- Contributing guidelines and changelog

### Changed
- Reorganized all code into proper package structure
- Updated all imports to use `llm_gen_ai` package
- Moved examples to dedicated `examples/` directory
- Moved experimental scripts to `experiments/` directory
- Moved documentation to `docs/` directory
- Moved notebooks to `notebooks/` directory
- Removed `requirements.txt` in favor of `pyproject.toml`
- Enhanced README with comprehensive documentation

## [0.1.0] - 2024-12-09

### Added
- Enhanced QA Agent with multi-stage accuracy pipeline
  - Query analysis and rewriting
  - Knowledge verification with confidence scoring
  - Parallel web search with time-sensitive filtering
  - Answer synthesis with citation
  - Automated answer refinement
  - Clarification handling for ambiguous queries
  - Human-in-the-loop feedback system
- Basic QA Agent with web search capability
- White paper generation agent
  - Multi-step planning
  - Research and validation
  - Context memory management
  - Document export to Word format
- Model quantization support for CPU inference
- Memory management and cache optimization
- Comprehensive module structure:
  - `query_analyzer` - Query analysis and rewriting
  - `knowledge_verifier` - Knowledge assessment
  - `parallel_search` - Concurrent web search
  - `answer_synthesizer` - Answer combination and citation
  - `clarification_handler` - Interactive query clarification
  - `answer_refiner` - Quality assessment and refinement
  - `feedback_loop` - Human feedback integration
- Support for multiple LLM models:
  - Llama 3.2 (3B, 7B)
  - Qwen 2.5 (3B)
  - Gemma 3 (1B)
- Example scripts and usage demonstrations
- Jupyter notebook for prompt guardrails
- Comprehensive documentation

### Changed
- Optimized model loading and caching
- Improved memory management
- Enhanced streaming support
- Better error handling and logging

### Fixed
- Memory overflow issues on MPS devices
- Infinite loop prevention in white paper generation
- Cache management for long conversations
- Token truncation issues

## Legend

- `Added` - New features
- `Changed` - Changes to existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Security vulnerability fixes

[Unreleased]: https://github.com/madhu1/llm-gen-ai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/madhu1/llm-gen-ai/releases/tag/v0.1.0
