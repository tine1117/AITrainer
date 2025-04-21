import json
from datasets import Dataset

# JSON 파일을 로드하고 변환하는 함수
def load_and_convert_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # JSON 데이터가 리스트 형식인지 확인
        if isinstance(data, list):
            processed_data = []
            for item in data:
                # instruction 필드가 없는 경우 기본값 설정
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output_text = item.get("output", "")
                
                # 리스트일 경우 문자열로 변환
                instruction = instruction[0] if isinstance(instruction, list) else instruction
                input_text = input_text[0] if isinstance(input_text, list) else input_text
                output_text = output_text[0] if isinstance(output_text, list) else output_text
                
                processed_data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                })
            return Dataset.from_list(processed_data)
        else:
            raise ValueError("JSON 데이터는 리스트 형식이어야 합니다.")
    
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"JSON 파일을 디코딩하는 중 오류가 발생했습니다: {file_path}")
        raise

# 여러 데이터셋을 결합하는 함수
def combine_datasets(*datasets):
    combined_data = []
    for dataset in datasets:
        for item in dataset:
            combined_data.append({
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", "")
            })
    return Dataset.from_list(combined_data)

# 훈련 데이터 준비 함수
def prepare_training_data(write_style_path, character_path, story_path):
    # 각 JSON 파일 로드
    write_style_dataset = load_and_convert_json(write_style_path)
    character_dataset = load_and_convert_json(character_path)
    story_dataset = load_and_convert_json(story_path)

    # 데이터셋 결합
    combined_dataset = combine_datasets(
        write_style_dataset,
        character_dataset,
        story_dataset
    )

    return combined_dataset