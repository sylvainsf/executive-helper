"""Quick test: does the merged safetensors model produce coherent output?
Isolates LoRA merge quality from GGUF conversion."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("models/executive-helper-ef")
print(f"  Vocab size: {tok.vocab_size}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "models/executive-helper-ef",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
print(f"  Model class: {model.__class__.__name__}")
print(f"  Model dtype: {model.dtype}")

messages = [
    {"role": "system", "content": "You are a compassionate, neuroaffirmative executive function support assistant."},
    {"role": "user", "content": "I need to clean my apartment but I can't start."},
]
inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
print(f"  Input tokens: {inputs.shape[1]}")
print(f"  Input text: {tok.decode(inputs[0])}")
print("---GENERATING---")
with torch.no_grad():
    out = model.generate(inputs, max_new_tokens=200, temperature=0.3, do_sample=True)
response = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"Response:\n{response}")
