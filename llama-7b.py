import accelerate
import llm
import time
from functools import wraps
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    # Load the LLaMA 7B model
    return AutoModelForCausalLM.from_pretrained(model_path)

@track_process_time
def load_tokenizer_model(model_path):
    return AutoTokenizer.from_pretrained(model_path)

@track_process_time    
def generate(input):
    #tokenize input
    model_inputs = tokenizer(
                            input, 
                            is_pretokenized=True, 
                            padding=True, 
                            truncation=True, 
                            return_tensors="pt")
    print(model_inputs)
    # Generate some text
    encoded_text = model.generate(**model_inputs, max_length=100)
    return tokenizer.decode(encoded_text)


model_path = "Llama-2-7b-hf"
model = load_preptrained_model(model_path)
tokenizer = load_tokenizer_model(model_path)

print(generate(["Once upon on a time"]))
