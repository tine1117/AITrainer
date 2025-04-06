import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("./storybook-model")
model = AutoModelForCausalLM.from_pretrained("./storybook-model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_story(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)