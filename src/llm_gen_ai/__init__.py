"""
llm-gen-ai: LLM-powered question answering and document generation agents.

This package provides AI agents for:
- General question answering with enhanced accuracy
- White paper and document generation
- Human-in-the-loop feedback systems
"""

__version__ = "0.1.0"

from llm_gen_ai.agents.basic_qa_agent import GeneralQAAgent
from llm_gen_ai.agents.enhanced_qa_agent import EnhancedQAAgent

__all__ = ["GeneralQAAgent", "EnhancedQAAgent"]
