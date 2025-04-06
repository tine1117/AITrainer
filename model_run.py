from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# device 설정 (GPU 사용 가능 시 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 저장된 모델 경로
model_path = "./server_model/storybook_model"

if not os.path.exists(model_path):
    print("모델이 존재하지 않습니다. 모델을 먼저 학습시켜 주세요.")
    exit()

# 모델 & 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# 모델이 저장된 경로
LOCAL_MODEL_PATH = "./server_model/storybook_model"

def generate_text(prompt):
    style_prompt = f"""
너는 어린이 동화작가야.
따뜻하고 부드러운 문체로 쓰고, 어린이가 이해하기 쉬운 단어만 사용해.
반드시 한국어로 작성해.
줄글 형태로 자연스럽게 이어서 써줘.

[동화 시작]
{prompt}
[동화 끝]
"""

    inputs = tokenizer(style_prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs["input_ids"],
        max_length=300,
        temperature=0.85,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.2,
        do_sample=True,
        num_beams=1
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 테스트
if __name__ == "__main__":

    if(not os.path.exists(LOCAL_MODEL_PATH)):
        print("모델이 존재하지 않습니다. 모델을 먼저 학습시켜 주세요.")
        exit()
    test_prompt = "작은 토끼가 숲속 친구들을 찾아 떠나는 이야기"
    generated_text = generate_text(test_prompt)

    print("생성된 텍스트:\n")
    print(generated_text)