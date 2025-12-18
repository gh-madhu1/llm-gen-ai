"""
Utility functions for the White Paper Generation Agent.
Contains shared helper functions and decorators.
"""
import time
import re
import os
from functools import wraps

from llm_gen_ai.config import ENABLE_TIMING_LOGS


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
