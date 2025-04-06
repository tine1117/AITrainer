# AITrainer
인공지능 학습 모듈(연습용)

![image](https://github.com/user-attachments/assets/2501d559-aada-431b-995e-48c11372ce9d)

HuggingFace 기반의 학습도구(transformer, peft, datasets)로 제작되었으며, 모델은 Qwen 2.5 버전, 크기는 0.5b 기반으로 FineTuning 작업을 하는 파이썬 코드임을 알립니다.
추후 학습에 필요한 변수는 변경 가능하도록 UI 혹은 yaml 파일로 바꿀 예정이며, 차후 데이터 모델이 충족된다면 Hugging Face으로 올릴 예정입니다.

**준비사항**
 - 아나콘다(권장)
 - python 3.12+

**가이드**
1. 다운로드 후 압축을 푼다.
2. 터미널을 통해 압축을 푼 폴더로 들어간다.
3. ``` pip install -f requirements.txt ``` 명령어를 이용해서 설치해 준다.
4. python *.py를 통해 파이썬 파일을 실행한다.

**requirements.txt** - 학습에 필요한 모듈과 API 구동 모듈을 모두 모아 놓은 파일입니다, 
가이드에 말한 것 처럼 ``` pip install -f requirements.txt ``` 명령어로 필요한 모듈을 모두 설치해서 쓸 수 있습니다.

**train.py** - 학습용 파이썬 파일, 이 파일 95줄 항목에서 맥 혹은 cpu용으로 돌릴려면 False 혹은 주석 필요 합니다.

**model_run.py** - 학습 완료시 모델을 구동 가능하게 하는 파이썬 모델

**data.py** - 학습에 필요한 목업 데이터 파일 

server_model 내 파일(베타)

**app.py** - flask를 이용한 API 가동 파일

**utils.py** - API 단에서 들어온 프롬프트를 모델으로 구동하고, 그 결과를 전달하는 파일
