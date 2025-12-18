"""
Core module for loading and managing LLM models.
"""
import os
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Try to import quantization config
try:
    from transformers import BitsAndBytesConfig
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False

from llm_gen_ai.config import (
    DEVICE, MODEL_PATH,
    USE_QUANTIZATION, QUANTIZATION_BITS,
    USE_TORCH_COMPILE, COMPILE_MODE
)

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
        torch_dtype=torch.float16 if DEVICE.type in ["cuda", "mps"] else torch.float32,
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


def clear_device_cache():
    """Clear GPU/MPS cache to free memory."""
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()
