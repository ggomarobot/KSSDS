# KSSDS: Korean Sentence Splitter for Dialogue Systems

[![PyPI version](https://badge.fury.io/py/KSSDS.svg)](https://pypi.org/project/KSSDS/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



### 한국어 대화 시스템 용 문장 분리기

[KSSDS](https://huggingface.co/ggomarobot/KSSDS) 는 딥러닝 기반의 한국어 문장 분리 모델입니다.

기존의 한국어 문장 분리기([KSS](https://github.com/hyunwoongko/kss), [Kiwi](https://github.com/bab2min/Kiwi) 등)는 규칙 또는 통계 기반의 모델로, 종결 어미나 구두점에 크게 의존하는 경향이 있습니다.  
이러한 특성 때문에, STT(Speech-to-Text) 모델을 통해 생성된 텍스트에서 자주 발생하는 변칙적인 사례(예: 구두점 생략, 어순 도치 등)에 효과적으로 대응하기 어려운 한계가 있습니다.

KSSDS는 이러한 한계를 극복하기 위해 개발된 모델로, 트랜스포머 기반 딥러닝을 활용하여 대화 시스템에서도 안정적이고 유연한 문장 분리를 목표로 합니다.

KSSDS는 HuggingFace Hub의 `lcw99/t5-base-korean-text-summary` [모델](https://huggingface.co/lcw99/t5-base-korean-text-summary)을 기반으로 하며,  
AI HUB에서 제공하는 다양한 음성 및 텍스트 데이터를 LLM을 활용해 개별 문장으로 분리(Pseudo-Label)한 후,  
각 문장의 끝 토큰을 종결 어미로 간주하여 token classification 모델로 Fine-Tuning하여 제작되었습니다.

자세한 과정은 이 [프레젠테이션 슬라이드](#)에서 확인하실 수 있습니다. 

---

## Table of Contents
1. [특징](#1-특징)
2. [설치 및 사용 방법](#2-설치-및-사용-방법)
3. [Folder Structure](#folder-structure)
4. [License](#license)
5. [Acknowledgments](#acknowledgments)

---

## 1. 특징

KSSDS는 한국어 대화 시스템 용 문장 분리에 특화된 딥러닝 기반 모델로,  
기존의 규칙 또는 통계 기반 문장 분리기들(KSS, Kiwi 등)과는 다음과 같은 차별화된 특징을 갖추고 있습니다:

1. **구두점 의존도 감소**  
    기존 문장 분리기는 주로 종결 어미나 구두점에 의존하여 문장을 분리합니다. 하지만 이러한 방식은 구두점이 없는 텍스트나 구두점이 문장의 끝이 아닌 경우에는 한계를 보일 수 있습니다.  
    KSSDS는 딥러닝 모델을 기반으로 이러한 의존도를 줄이고, 구두점 유무와 관계없이 보다 자연스러운 문장 분리가 가능합니다.

2. **반복되는 단어 처리**  
    특정 단어나 구문이 반복적으로 나타날 경우 이를 자동으로 감지하고 문장으로 분리하는 기능을 제공합니다. OpenAI Whisper와 같은 STT 모델은 충분한 정보를 얻지 못할 경우, 이전에 생성된 단어나 구문을 반복적으로 출력하는 경향이 있습니다[^1]. Whisper는 텍스트 생성 시 마지막 224개의 토큰만을 참고하여 다음 단어를 예측하는데[^2], 이로 인해 충분한 정보를 확보하지 못할 경우 동일한 단어가 최대 223번 반복되는 현상이 자주 관찰됩니다.

    이러한 반복 구문은 기존의 규칙 또는 통계 기반 문장 분리기에서는 사전적 의미의 "문장"으로 간주되지 않아 처리가 어렵습니다.
    그러나 KSSDS는 문장 분리 후 추가적인 Rule을 적용하여 비정상적으로 긴 반복 구문을 감지하고, 이를 적절히 분리함으로써 후속 NLP 모델의 입력 길이 제한을 초과하지 않도록 최적화된 결과를 제공합니다.

3. **딥러닝 기반의 문장 분리**  
    KSSDS는 T5 모델의 Encoder를 Fine-Tuning하여 학습된 모델로, 어순 도치와 같은 복잡한 문장 구조에서도 기존의 KSS 또는 Kiwi보다 우수한 성능을 보이는 경우가 많습니다. 단순히 종결 어미에 의존하기보다는, 문장의 의미를 기반으로 한 문장 분리 방식을 제공합니다.

자세한 문장 분리 결과는 이 [Google Colab notebook](https://colab.research.google.com/drive/1gudvMJ4NDwk-V6Qh1VP2g3kL5QRicfc4?usp=sharing)에서 확인 가능합니다.

---

> **Note**: 
> KSSDS는 연구 목적으로 개인이 개발한 초기 모델로, 학습 데이터의 한계로 인해 일부 데이터셋에서는 성능 차이가 발생할 수 있습니다. 

> 아직 부족한 부분이 많기에 개발자분들의 많은 관심과 기여 부탁드립니다.


## 2. 설치 및 사용 방법

KSSDS는 GitHub에서 코드를 직접 받아 사용하는 방법과, PyPI를 통해 패키지를 설치하여 사용하는 두 가지 방법을 제공합니다.  
각각의 설치 및 사용 방법은 아래를 참고하세요.

### 2.1 GitHub에서 설치하기

GitHub 저장소에서 코드를 다운로드하면, 추가 학습, 평가, 추론 등 다양한 작업을 지원하는 스크립트를 사용할 수 있습니다.

#### 설치
1. GitHub 저장소를 클론합니다.
   ```bash
   git clone https://github.com/ggomarobot/KSSDS.git
   cd KSSDS
   ```
2. 의존성을 설치합니다.
   ```bash
   pip install -r requirements.txt    
   ```

#### 사용

- **추가 학습**: `./run.sh` 를 실행하여 학습을 수행합니다.
    ```bash
    ./run.sh
    ```
    - 기본적으로 `config/main_config.yaml` 파일의 설정을 따릅니다.
    - 필요한 경우 `main_config.yaml`을 수정하여 학습 데이터 경로, 모델 저장 경로, 배치 크기 등을 조정하세요.

- **평가**: `./evaluate.sh` 를 실행하여 학습된 모델의 성능을 평가합니다.
    ```bash
    ./evaluate.sh
    ```
    - 평가 설정은 `config/eval_config.yaml` 파일에서 제어할 수 있습니다.

- **추론**: `./inference.sh` 를 실행하여 문장 분리를 수행합니다.
    ```bash
    ./inference.sh
    ```
    - `config/inference_config.yaml` 파일을 사용하여 입력 데이터 경로 및 출력 형식을 설정할 수 있습니다.

### 2.2 PyPI에서 설치하기

PyPI를 통해 KSSDS 패키지를 설치하고 사용할 수 있습니다. 이 경우, 간단한 문장 분리 기능만 제공되며, 추가 학습이나 평가 기능은 지원되지 않습니다.

#### 설치

아래 명령어를 사용하여 KSSDS를 설치할 수 있습니다:

```bash
pip install KSSDS
```

#### 사용 방법

설치 후, KSSDS는 Python 코드에서 다음과 같이 사용할 수 있습니다:

```python
from KSSDS import KSSDS

# KSSDS 클래스 초기화
kssds = KSSDS()

# 입력 텍스트
input_text = "안녕하세요. 오늘 날씨가 참 좋네요. 저는 산책을 나갈 예정입니다."

# 문장 분리 실행
split_sentences = kssds.split_sentences(input_text)

# 결과 출력
for idx, sentence in enumerate(split_sentences):
    print(f"{idx + 1}: {sentence}")
```
#### 출력 예시
<pre style="background-color:#F5EDED; color:white; padding:10px; border-radius:5px; font-family:monospace;">
<span style="color:#a29acb;">1: 안녕하세요.</span>
<span style="color:#c3adad;">2: 오늘 날씨가 참 좋네요.</span>
<span style="color:YellowGreen;">3: 저는 산책을 나갈 예정입니다.</span>
</pre>

> **Note: PyPI 패키지로 설치한 KSSDS는 추가적인 설정 파일(YAML) 없이 기본 설정값을 사용하여 동작합니다.**

---

## Folder Structure
```
.
├── config                   # Configuration files (YAML)
├── data                     # Training and evaluation data (excluded from the repo)
├── notebooks                # Jupyter notebooks for analysis and experiments
├── src                      # Source code
│   ├── KSSDS                # KSSDS package modules
│   │   ├── __init__.py      # Package initializer
│   │   ├── inference.py     # Main inference logic
│   │   ├── dataloader.py    # Custom dataloaders
│   │   └── dataset.py       # Dataset handling
├── tests                    # Unit tests
├── LICENSE                  # License information
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
└── setup.py                 # Installation script

. 
├── config    # Configuration files (YAML) for training, evaluation, and inference 
├── src/KSSDS # Source codes for KSSDS package modules
├── tests     # Unit tests
├── data      # Placeholder for datasets (see the README inside for Google Drive link) 
├── models    # Placeholder for KSSDS models (see the README inside for HF Hub link)  
├── notebooks # Jupyter notebooks for analysis
│ 
├── env.sh # Script to set up the environment variables 
├── evaluate.sh # Shell script to run evaluation 
├── inference.sh # Shell script to perform inference 
├── requirements.txt # List of Python dependencies 
├── run.sh # Shell script to start training 
├── setup.py # Python setup file for packaging the project 
└── tests # Unit tests and examples for the package
```

### Notes:
- **`data`**: This folder is excluded from the repository due to its size. Refer to the README inside the `data` folder for instructions and a Google Drive link.
- **`config`**: The YAML files allow users to customize the behavior of training, evaluation, and inference.
- **`notebooks`**: These provide step-by-step usage examples, including comparisons with other models like Kiwi and KSS.
- **`src/KSSDS`**: Contains the main code for the KSSDS package, including the inference module.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this project, provided that proper credit is given to the original author. See the [LICENSE](LICENSE) file for more details.


[^1]:[Whisper word repeat issue](https://github.com/jhj0517/Whisper-WebUI/issues/238)

[^2]: [OpenAI Whisper Discussion - Token Limit](https://github.com/openai/whisper/discussions/1824#discussioncomment-7620322)