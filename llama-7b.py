import accelerate
import llm
import time
from functools import wraps
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

device = torch.device("cpu")

# Trakcs the process execution time 
def track_process_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.process_time()
        result = func(*args, **kwargs)
        end_time = time.process_time()
        process_time = end_time - start_time
        print(f"Process time for {func.__name__}: {process_time:.3f} seconds")
        return result
    return wrapper
    
@track_process_time
def load_preptrained_model(model_path):
    # Load the pre-trained LLAMA-7B model.
    model = AutoModelForCausalLM.from_pretrained("Llama-2-7b-hf", 
                            quantization_config=BitsAndBytesConfig(
                                            load_in_8bit_int8_cpu=True))
    return model

@track_process_time
def load_tokenizer_model(model_path):
    return AutoTokenizer.from_pretrained('Llama-2-7b-hf',
                            quantization_config=BitsAndBytesConfig(
                                            load_in_8bit_int8_cpu=True))

@track_process_time    
def generate(input):
    generated_text = model.generate(input_ids=tokenizer(input, return_tensors="pt").input_ids, max_length=100)
    try:
        return tokenizer.decode(generated_text[0])
    except:
        return 'Generated text decode error'
    
    return None

model_path = "Llama-2-7b-hf"
model = load_preptrained_model(model_path)
tokenizer = load_tokenizer_model(model_path)

print(generate("Once upon on a time there was a king"))
