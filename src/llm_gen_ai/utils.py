"""
Utility functions for the White Paper Generation Agent.
Contains shared helper functions and decorators.
"""
import time
import re
import os
from functools import wraps, lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Try to import quantization config, but don't fail if not available (Mac)
try:
    from transformers import BitsAndBytesConfig
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False

from llm_gen_ai.config import (
    DEVICE, MODEL_PATH, ENABLE_TIMING_LOGS,
    USE_QUANTIZATION, QUANTIZATION_BITS,
    USE_TORCH_COMPILE, COMPILE_MODE
)


def track_process_time(func):
    """Decorator to track and print function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.process_time()
        result = func(*args, **kwargs)
        end_time = time.process_time()
        process_time = end_time - start_time
        if ENABLE_TIMING_LOGS:
            print(f"⏱️  Process time for {func.__name__}: {process_time:.3f} seconds")
        return result
    return wrapper


@lru_cache(maxsize=1)
def load_pretrained_model(model_path=MODEL_PATH):
    """
    Load the pre-trained model onto the selected device.
    Uses LRU cache to avoid reloading.
    OPTIMIZED: Supports quantization and torch.compile for faster inference.
    """
    if USE_QUANTIZATION:
        return load_quantized_model(model_path)
    
    print(f"Loading model from {model_path} on {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16 if DEVICE.type in ["cuda", "mps"] else torch.float32,
        low_cpu_mem_usage=True  # Optimize memory during loading
    )
    model.eval()  # Set to evaluation mode for inference
    
    # Apply torch.compile for additional speedup
    if USE_TORCH_COMPILE:
        model = compile_model(model)
    
    return model


@lru_cache(maxsize=1)
def load_quantized_model(model_path=MODEL_PATH):
    """
    Load model with quantization for faster inference and lower memory usage.
    4-bit quantization: ~4x memory reduction, 2-3x speedup
    8-bit quantization: ~2x memory reduction, 1.5-2x speedup
    """
    # Check if quantization is supported
    if not HAS_QUANTIZATION:
        print("⚠️  BitsAndBytes not available. Falling back to FP16 model.")
        return _load_fp16_model(model_path)
    
    if DEVICE.type not in ["cuda"]:
        print(f"⚠️  Quantization not supported on {DEVICE.type}. Falling back to FP16 + torch.compile.")
        return _load_fp16_model(model_path)
    
    print(f"Loading {QUANTIZATION_BITS}-bit quantized model from {model_path}...")
    
    # Configure quantization
    if QUANTIZATION_BITS == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Nested quantization for better accuracy
            bnb_4bit_quant_type="nf4"  # Normal Float 4-bit
        )
    elif QUANTIZATION_BITS == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    else:
        raise ValueError(f"Unsupported quantization bits: {QUANTIZATION_BITS}. Use 4 or 8.")
    
    # Load with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    print(f"✅ Model loaded with {QUANTIZATION_BITS}-bit quantization")
    
    return model


def _load_fp16_model(model_path):
    """Helper to load FP16 model with torch.compile."""
    print(f"Loading FP16 model from {model_path} on {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE.type in ["cuda", "mps"] else torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # Apply torch.compile for speedup on Mac MPS
    if USE_TORCH_COMPILE:
        model = compile_model(model)
    
    return model


def compile_model(model):
    """
    Compile model using torch.compile for faster inference.
    Provides 2-3x speedup on compatible hardware.
    """
    try:
        print(f"Compiling model with mode='{COMPILE_MODE}'...")
        compiled_model = torch.compile(model, mode=COMPILE_MODE)
        print("✅ Model compiled successfully")
        return compiled_model
    except Exception as e:
        print(f"⚠️  Model compilation failed: {e}. Using uncompiled model.")
        return model


@lru_cache(maxsize=1)
def load_tokenizer_model(model_path=MODEL_PATH):
    """
    Load the tokenizer for the model.
    Uses LRU cache to avoid reloading.
    """
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def sanitize_filename(filename, max_length=50):
    """
    Sanitize filename to prevent errors and limit length.
    
    Args:
        filename: Original filename
        max_length: Maximum length for base name (default: 50)
    
    Returns:
        Sanitized filename
    """
    filename = filename.strip()
    # Remove path separators and other problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit filename length (excluding extension)
    base_name = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1] if '.' in filename else '.docx'
    if len(base_name) > max_length:
        base_name = base_name[:max_length]
    return f"{base_name}{extension}"


def clean_content(text):
    """
    Remove agent thoughts and actions from content.
    
    Args:
        text: Raw text with potential agent markers
    
    Returns:
        Cleaned text without agent markers
    """
    # Remove THOUGHT, ACTION, OBSERVATION patterns
    text = re.sub(r'THOUGHT:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'ACTION:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'OBSERVATION:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()


def clear_device_cache():
    """Clear GPU/MPS cache to free memory."""
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def truncate_text(text, max_length, keep_prefix=1000, keep_suffix=3000):
    """
    Intelligently truncate text to fit within max_length.
    Keeps important parts from beginning and end.
    
    Args:
        text: Text to truncate
        max_length: Maximum length in characters
        keep_prefix: Characters to keep from start
        keep_suffix: Characters to keep from end
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    if keep_prefix + keep_suffix >= max_length:
        # Adjust proportionally
        ratio = max_length / (keep_prefix + keep_suffix)
        keep_prefix = int(keep_prefix * ratio)
        keep_suffix = int(keep_suffix * ratio)
    
    return text[:keep_prefix] + "\n...[truncated]...\n" + text[-keep_suffix:]


def count_tokens(text, tokenizer):
    """
    Count the number of tokens in text.
    
    Args:
        text: Text to count
        tokenizer: Tokenizer to use
    
    Returns:
        Number of tokens
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
