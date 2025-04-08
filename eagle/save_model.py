from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify a local directory to save the model and tokenizer
local_model_path = "./qwen2.5-7b-instruct-local"

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.save_pretrained(local_model_path)

# Download and save the model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model.save_pretrained(local_model_path)

print(f"Model and tokenizer saved to {local_model_path}")