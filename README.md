# KSSDS: Korean Sentence Splitter for Dialogue Systems

[![PyPI version](https://badge.fury.io/py/KSSDS.svg)](https://pypi.org/project/KSSDS/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



### 한국어 대화 시스템 용 문장 분리기

[KSSDS](https://huggingface.co/ggomarobot/KSSDS)는 딥러닝 기반 한국어 문장 분리 모델입니다.  

[KSS](https://github.com/hyunwoongko/kss)나 [Kiwi](https://github.com/bab2min/Kiwi) 같은 기존의 한국어 문장 분리기들은 규칙 또는 통계 기반 모델로, 종결 어미나 구두점에 의존적인 경우가 많습니다. 이러한 특성으로 인해, STT(Speech-to-Text) 모델을 통해 생성된 텍스트에서 발생할 수 있는 다양한 변칙적인 사례(예: 구두점 생략, 도치 화법 등)에 유연하게 대처하기 어렵습니다.  

이러한 한계를 극복하고자, 트랜스포머 기반의 딥러닝 모델을 활용하여 대화 시스템에서 기존 모델들보다 정확한 문장 분리 성능을 목표로 개발하게 되었습니다.  

KSSDS는 HuggingFace Hub의 `lcw99/t5-base-korean-text-summary` 모델[^1]을 기반으로, AI HUB에서 제공하는 다양한 음성 및 텍스트 데이터를 LLM을 이용해 개별 문장으로 분리한 뒤, 각 문장의 끝 토큰을 종결 어미로 간주(Pseudo-Label)하여 token classification 모델로 Fine-Tuning 한 것입니다.

자세한 과정은 이 [프레젠테이션 슬라이드](#)에서 확인하실 수 있습니다. 

---

## Table of Contents
1. [특징](#특징)
2. [설치](#설치)
3. [Usage](#usage)
4. [Features](#features)
5. [Folder Structure](#folder-structure)
6. [License](#license)
7. [Acknowledgments](#acknowledgments)

---

## 1. 특징

KSSDS는 한국어 대화체 문장 분리에 특화된 딥러닝 기반의 모델로, 기존의 규칙 기반 또는 통계 기반 문장 분리기(KSS, Kiwi 등)와는 차별화된 다음과 같은 특징을 가지고 있습니다:

1. **구두점 의존도 감소**  
   기존 문장 분리기들은 종결 어미나 구두점에 의존하여 문장을 분리하는 경우가 많습니다. KSSDS는 딥러닝 모델을 활용하여 이러한 의존도를 줄이고, 구두점이 없는 텍스트나 구두점이 문장의 끝이 아닌 경우에도 비교적 자연스러운 문장 분리가 가능합니다.

| Original Text                              | KSS                              | Kiwi                             | KSSDS                            |
|--------------------------------------------|-----------------------------------|----------------------------------|----------------------------------|
| 여기 혹시 찹쌀도나쓰? 찹쌀도너츠? 라는 거 팔아요? | 여기 혹시 찹쌀도나쓰?           | 여기 혹시 찹쌀도나쓰?           | 여기 혹시 찹쌀도나쓰?  찹쌀도너츠? 라는 거 팔아요?         |
|                                            | 찹쌀도너츠? 라는 거 팔아요?                       | 찹쌀도너츠?                      |      |
|                                            |                    | 라는 거 팔아요?                  |                                  |

| Original Text                                                                                                                                       | KSS                              | Kiwi                             | KSSDS                            |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|----------------------------------|----------------------------------|
| 우리가 타이라는 단어는 원래 동사로 묶다. 엮다. 라는 표현이고요. 그다음에 명사로 하게 되면 묶음이라는 표현이죠. | 우리가 타이라는 단어는 원래 동사로 묶다. | 우리가 타이라는 단어는 원래 동사로 묶다. | 우리가 타이라는 단어는 원래 동사로 묶다. 엮다. 라는 표현이고요. 그다음에 명사로 하게 되면 묶음이라는 표현이죠.|
|                                                                                                                                                       | 엮다. 라는 표현이고요              | 엮다.            |                      |
|                                                                                                                                                       | 그다음에 명사로 하게 되면 묶음이라는 표현이죠. | 라는 표현이고요. 그다음에 명사로 하게 되면 묶음이라는 표현이죠.       |                                  |



2. **반복되는 단어 처리**  
   동일한 단어가 반복적으로 나타나는 경우 이를 자동으로 감지하고 문장으로 분리할 수 있는 기능을 제공합니다. OpenAI Whisper와 같은 STT 모델은 특정 상황에서 충분한 정보를 얻지 못할 경우, 이전에 생성된 문장이나 단어를 반복적으로 출력하는 경향이 있습니다. Whisper는 텍스트를 생성할 때, 이전에 생성된 텍스트 중 마지막 224개의 토큰만을 참고하여 다음 단어를 예측합니다[^2]. 이 과정에서 충분한 정보를 얻지 못하면, 현재 참고 중인 224개의 토큰을 전부 넘길 때까지 동일한 단어가 최대 223번 반복되는 현상이 자주 관찰됩니다.

    이와 같은 반복 구간은 기존의 규칙 기반 또는 통계 기반 문장 분리기에서는 사전적 의미의 "문장"으로 간주되지 않아 처리가 어려울 수 있습니다. 그러나 KSSDS는 문장 분리 이후 추가 Rule을 적용하여 특정 길이 이상의 반복 구문을 감지하고 이를 자동으로 분리함으로써, 후속 NLP 모델이 입력 길이 제한을 초과하지 않도록 최적화된 결과를 제공합니다.

| **Original Text** | **KSS** | **Kiwi** | **KSSDS** |
|--------------------|---------|----------|-----------|
| 근데 지금 와서 보니까 아니잖아. 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 추리하다가 옆집에 피해주는 거 있잖아요 시끄럽게 그리고 또 쉽게 말해서 우리 손주는 동적이지 않고 정적인 애라서 뛰거나 막 이러지 않아요 | 근데 지금 와서 보니까 아니잖아. | 근데 지금 와서 보니까 아니잖아. | 근데 지금 와서 보니까 아니잖아. |
|  | 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 추리하다가 옆집에 피해주는 거 있잖아요 | 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 추리하다가 옆집에 피해주는 거 있잖아요 | 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 |
|  | 시끄럽게 그리고 또 쉽게 말해서 우리 손주는 동적이지 않고 정적인 애라서 뛰거나 막 이러지 않아요 | 시끄럽게 그리고 또 쉽게 말해서 우리 손주는 동적이지 않고 정적인 애라서 뛰거나 막 이러지 않아요 | 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 |
|  |  |  | 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 |
|  |  |  | 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 아 |
|  |  |  | 추리하다가 옆집에 피해주는 거 있잖아요 |
|  |  |  | 시끄럽게 그리고 또 쉽게 말해서 우리 손주는 동적이지 않고 정적인 애라서 뛰거나 막 이러지 않아요 |


3. **딥러닝 기반의 문장 분리**  
   KSSDS는 T5 모델의 Encoder를 Fine-Tuning하여 학습되었으며, 어순 도치와 같은 변칙적인 입력에서 KSS 또는 Kiwi보다 나은 성능을 보이는 경우가 있습니다. 단순한 "종결 어미"에 의존하기보다, 문장의 의미를 기반으로 한 분리 방식을 제공합니다.

| **Original Text** | **KSS** | **Kiwi** | **KSSDS** |
|--------------------|---------|----------|-----------|
| 아 이미 거기 다녀왔어 우리는 | 아 이미 거기 다녀왔어 | 아 이미 거기 다녀왔어 | 아 이미 거기 다녀왔어 우리는 |
|  | 우리는 | 우리는 |  |

---

> **Note**: 
> KSSDS는 연구 목적으로 개인이 개발한 초기 모델로, 학습 데이터의 한계로 인해 일부 데이터셋에서는 성능 차이가 발생할 수 있습니다. 

> 아직 부족한 부분이 많기에 개발자분들의 많은 관심과 기여 부탁드립니다.


## 설치

### Using PyPI
Install KSSDS via pip:
```bash
pip install KSSDS
```

### Clone the Repository

Alternatively, clone the GitHub repository:
```bash
git clone https://github.com/ggomarobot/KSSDS.git
cd KSSDS
pip install -e .
```

---

## Usage

### Inference Example

#### From Python Script:


```python
from KSSDS import KSSDS

# Initialize the model
kssds = KSSDS()

# Split sentences
input_text = "안녕하세요. 오늘 날씨가 참 좋네요. 저는 산책을 나갈 예정입니다."
split_sentences = kssds.split_sentences(input_text)

# Print results
for idx, sentence in enumerate(split_sentences):
    print(f"{idx + 1}: {sentence}")
```

<pre style="background-color:#F5EDED; color:white; padding:10px; border-radius:5px; font-family:monospace;">
<span style="color:#a29acb;">1: 안녕하세요.</span>
<span style="color:#c3adad;">2: 오늘 날씨가 참 좋네요.</span>
<span style="color:YellowGreen;">3: 저는 산책을 나갈 예정입니다.</span>
</pre>

#### From Terminal:

```bash
python ./src/inference.py --config_path ./config/inference_config.yaml
```

## Features
- **Repetition Detection**: Handles STT hallucinations by detecting and splitting repetitive patterns.

- **High Accuracy**: Fine-tuned on diverse datasets to ensure precise sentence splitting.

- **Configurable**: Adjust parameters such as repetition thresholds via YAML configuration files.

- **Multi-Platform Support**: Compatible with both text and TSV inputs.

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
```

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this project, provided that proper credit is given to the original author. See the [LICENSE](LICENSE) file for more details.


[^1]:[lcw99/t5-base-korean-text-summary 모델](https://huggingface.co/lcw99/t5-base-korean-text-summary)

[^2]: [OpenAI Whisper Discussion - Token Limit](https://github.com/openai/whisper/discussions/1824#discussioncomment-7620322)