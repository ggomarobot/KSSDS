# Models 폴더 안내

KSSDS 모델과 관련된 두 가지 HuggingFace Hub 링크를 제공합니다. 아래 링크를 통해 모델을 확인하고 활용할 수 있습니다:

## HuggingFace Hub 모델 링크

- [**KSSDS**](https://huggingface.co/ggomarobot/KSSDS)  
  - 학습 데이터셋에서 적절한 Length Filter를 적용하여 학습된 KSSDS의 공식 모델입니다.  
  - Whisper와 같은 STT 모델에서 생성된 transcription을 보다 정확하게 분리할 수 있도록 설계되었습니다.

- [**KSSDS_NO_LF**](https://huggingface.co/ggomarobot/KSSDS_NO_LF)  
  - Length Filter를 적용하지 않은 ablation study 용 모델입니다.  
  - 학습 데이터 필터링을 생략하여, 긴 텍스트에서 상대적으로 덜 정밀한 문장 분리를 수행합니다.  
  - 연구 및 비교 목적 외에는 실제 사용에 적합하지 않을 수 있습니다.


## KSSDS 모델 사용법

KSSDS를 체험하거나 활용할 수 있는 방법은 다음 세 가지가 있습니다:

1. **PyPI 설치 (권장)**  
   PyPI를 통해 KSSDS를 설치하여 문장 분리 기능을 빠르게 활용할 수 있습니다.  
   PyPI 설치와 관련된 자세한 내용은 [루트 디렉토리 README의 PyPI 설치 섹션](../README.md#32-pypi에서-설치하기)을 참조하세요.

2. **GitHub 설치**  
   추가 학습, 평가, 고급 추론 설정 등을 원하시는 개발자분들은 GitHub 저장소를 클론하여 다양한 기능을 활용할 수 있습니다.  
   GitHub 설치와 관련된 자세한 내용은 [루트 디렉토리 README의 GitHub 설치 섹션](../README.md#31-github에서-설치하기)을 참조하세요.

3. **Hugging Face Hub**  
   Hugging Face Hub의 네이티브 방식을 통해 KSSDS를 경험할 수 있습니다.  
   T5 Encoder와 모델 관련 설정 파일을 동적으로 다운로드하여 사용할 수 있으며, [Hugging Face Hub 모델 사용법](https://huggingface.co/ggomarobot/KSSDS)을 참조하세요.

> 각 방식의 장단점은 다음과 같습니다:  
> - PyPI: 간편한 설치 및 사용, 추론 기능만 지원  
> - GitHub: 추론 기능 외에도 추가적인 학습 및 평가 가능  
> - Hugging Face Hub: 특정 환경에서 빠르게 모델 활용 가능   

아래는 PyPI 및 GitHub 방식으로 설치 후 사용하는 방법에 대한 예제입니다:

### 기본 사용법

```python
from KSSDS import KSSDS

# 기본 설정으로 KSSDS 모델 호출
kssds = KSSDS()

# 입력 텍스트
input_text = "안녕하세요. 오늘 날씨가 참 좋네요. 저는 산책을 나갈 예정입니다."

# 문장 분리 실행
split_sentences = kssds.split_sentences(input_text)

# 결과 출력
for idx, sentence in enumerate(split_sentences):
    print(f"{idx + 1}: {sentence}")
```

## 사용자 설정 가능 파라미터

`KSSDS` 클래스의 `__init__` 메서드는 다음과 같은 파라미터를 제공합니다:

| 파라미터              | 설명                                                                                     | 기본 값                          |
|-----------------------|----------------------------------------------------------------------------------------|----------------------------------|
| `config_path`         | YAML 설정 파일 경로                                                                     | `None` (기본 설정 사용)         |
| `model_path`          | HuggingFace Hub에서 불러올 모델 경로                                                     | `"ggomarobot/KSSDS"`            |
| `tokenizer_path`      | HuggingFace Hub에서 불러올 토크나이저 경로                                               | `"ggomarobot/KSSDS"`            |
| `max_repeats`         | 반복 단어 처리 시 한 문장으로 간주할 최대 반복 단어 수                                   | `60`                             |
| `detection_threshold` | 반복 단어를 감지하기 위한 최소 단어 수                                                | `70`                             |
| `max_phrase_length` | 반복 구절을 감지하기 위한 구절 내 최대 단어 수                                               | `2`                            |

`config_path`로 YAML 설정 파일을 전달하거나, 직접 파라미터를 지정할 수 있습니다.

## KSSDS_NO_LF 모델 사용법

Ablation Study 용으로 학습된 **KSSDS_NO_LF** 모델을 사용하려면 `model_path`와 `tokenizer_path`를 아래와 같이 설정하세요:

```python
from KSSDS import KSSDS

# KSSDS_NO_LF 모델 초기화
kssds_nolen = KSSDS(
    model_path="ggomarobot/KSSDS_NO_LF",
    tokenizer_path="ggomarobot/KSSDS_NO_LF"
)
```


## 파라미터 변경 예시
`KSSDS` 클래스의 다양한 파라미터를 사용자 정의할 수 있습니다. 예를 들어, 반복 단어 감지 기준을 변경하려면 다음과 같이 사용할 수 있습니다:

```python
from KSSDS import KSSDS

# 사용자 정의 파라미터로 KSSDS 초기화
kssds_custom = KSSDS(
    max_repeats=50,  # 반복 단어를 50개 단위로 분리
    detection_threshold=100  # 반복 감지를 위한 텍스트의 최소 단어 수를 100으로 설정
)
```

이렇게 설정을 통해 사용자는 자신만의 데이터 및 환경에 최적화된 문장 분리를 수행할 수 있습니다.
