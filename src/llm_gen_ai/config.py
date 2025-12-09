"""
Configuration module for the White Paper Generation Agent.
Centralizes all tunable parameters for easy optimization.
"""
import torch

# Device Configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Model Configuration
MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"
MAX_MODEL_LENGTH = 4096  # Maximum input tokens

# ===== INFERENCE OPTIMIZATION SETTINGS =====
# Quantization - Reduces memory and increases speed by 2-4x
# NOTE: Mac MPS does not support quantization - auto-disabled below
USE_QUANTIZATION = True if DEVICE.type in ["cuda"] else False  # Only on CUDA GPUs
QUANTIZATION_BITS = 4  # Options: 4 or 8 (4 = faster, 8 = more accurate)

# Torch Compile - Provides 2-3x speedup on compatible hardware
# This is the primary optimization for Mac MPS
USE_TORCH_COMPILE = True if DEVICE.type in ["mps", "cuda"] else False
COMPILE_MODE = "reduce-overhead"  # Options: "default", "reduce-overhead", "max-autotune"

# Generation Parameters - OPTIMIZED FOR SPEED AND QUALITY
GENERATION_CONFIG = {
    "do_sample": False,  # Greedy decoding for speed (deterministic)
    "temperature": 1.0,  # Not used when do_sample=False
    "top_p": 1.0,  # Not used when do_sample=False
    "repetition_penalty": 1.1,  # Prevent repetition
    "use_cache": True,  # Enable KV cache for speed - CRITICAL
}

# Alternative: Sampling mode (slower but more creative)
GENERATION_CONFIG_SAMPLING = {
    "do_sample": True,
    "temperature": 0.3,  # Lower = more deterministic and faster
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "use_cache": True,
}

# Default generation tokens (can be overridden per call)
DEFAULT_MAX_NEW_TOKENS = 100

# Agent Configuration
MAX_STEPS = 200  # Maximum agent steps before timeout
MAX_ACTIONS_MEMORY = 10  # Number of recent actions to track
MINIMUM_SECTIONS = 5  # Minimum sections before allowing FINISH

# Context Memory Configuration
MAX_PROMPT_LENGTH_CHARS = 8000  # Max characters before truncation
MAX_PROMPT_LENGTH_FOR_REBUILD = 3000  # Trigger context compression at this size
RESEARCH_SUMMARY_LENGTH = 200  # Characters to keep per research query

# Search Configuration
MAX_SEARCH_RESULTS = 3  # Results per search query
VALIDATION_SEARCH_RESULTS = 5  # Results for novelty validation
SEARCH_CACHE_ENABLED = True  # Enable caching to avoid duplicate searches

# Memory Management - Optimized for Inference
CLEAR_CACHE_FREQUENCY = "never"  # Options: "always", "never", "adaptive" (never = fastest)
USE_GRADIENT_CHECKPOINTING = False  # Disabled for inference (training only)

# Document Generation
OUTPUT_DIR = "."
DEFAULT_FILENAME = "white_paper.docx"
MAX_FILENAME_LENGTH = 50

# Standard white paper section order
SECTION_ORDER = [
    'title', 'author', 'abstract', 'executive summary', 'introduction',
    'background', 'problem statement', 'methodology', 'solution',
    'implementation', 'results', 'benefits', 'challenges',
    'recommendations', 'conclusion', 'references', 'appendix'
]

# Required sections for a complete white paper
REQUIRED_SECTIONS = {
    'title', 'abstract', 'introduction',
    'solution', 'benefits', 'challenges', 'conclusion'
}

# Performance Tuning Flags
ENABLE_TIMING_LOGS = True  # Print timing information
ENABLE_PROGRESS_LOGS = True  # Print progress updates
ENABLE_MEMORY_OPTIMIZATION = True  # Enable all memory optimizations
