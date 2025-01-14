# Configuration Files

이 폴더에는 KSSDS 모델의 **학습**, **평가**, **추론**을 제어할 수 있는 YAML 설정 파일들이 포함되어 있습니다.  
사용자는 각 파일을 수정하여 데이터 경로, 모델 파라미터, 로깅 옵션 등을 원하는 대로 조정할 수 있습니다.

## 파일 설명

- **`main_config.yaml`**: 학습 시 사용되는 설정 파일입니다. 모델의 학습 데이터 경로, 배치 크기, 학습 epoch 수 등 다양한 설정이 포함되어 있습니다.  
  실행 명령: `./run.sh`

- **`eval_config.yaml`**: 모델의 성능 평가 시 사용되는 설정 파일입니다. 테스트 데이터 경로, 평가 배치 크기, 모델 경로 등을 설정합니다.  
  실행 명령: `./evaluate.sh`

- **`inference_config.yaml`**: 문장 분리 추론을 실행할 때 사용되는 설정 파일입니다. 모델 경로, 입력 데이터 및 출력 경로, 반복 단어 처리 옵션 등을 제어합니다.  
  실행 명령: `./inference.sh`

## 주요 파라미터

### 1. `main_config.yaml`
#### 데이터 관련
- **`training_dataset_folder`**: 학습 데이터가 저장된 폴더 경로
- **`validation_dataset_folder`**: 검증 데이터 폴더 경로
- **`test_dataset_folder`**: 테스트 데이터 폴더 경로
- **`src_file_extension`**: 데이터 파일의 확장자 (예: `.tsv`)

#### 모델 관련
- **`model_path`**: 사전 학습된 모델 경로

#### 학습 설정
- **`per_device_train_batch_size`**: 학습 시 사용되는 배치 크기
- **`per_device_eval_batch_size`**: 평가 시 사용되는 배치 크기
- **`num_train_epochs`**: 학습 epoch 수
- **`logging_dir`**: 로그 파일이 저장될 경로
- **`early_stopping`**: 학습 조기 종료 옵션

---

### 2. `eval_config.yaml`
#### 데이터 관련
- **`test_dataset_folder`**: 평가에 사용할 테스트 데이터 폴더 경로
- **`src_file_extension`**: 데이터 파일의 확장자 (예: `.tsv`)

#### 모델 관련
- **`model_path`**: 평가에 사용할 모델 경로
- **`tokenizer_path`**: 토크나이저 경로

#### 평가 설정
- **`per_device_eval_batch_size`**: 평가 시 배치 크기
- **`logging_dir`**: 로그 파일 저장 경로

---

### 3. `inference_config.yaml`
#### 입력 및 출력 관련
- **`input_tsv`**: 입력 파일 경로 (TSV 파일)
- **`input_sequence`**: 입력 텍스트 문자열
- **`output_tsv`**: 결과를 저장할 출력 TSV 파일 경로
- **`output_print`**: 결과를 터미널에 출력 여부

#### 반복 단어 처리
- **`max_repeats`**: 한 문장으로 간주할 최대 반복 단어 수 (기본값: 60)
- **`detection_threshold`**: 반복 단어를 감지하기 위한 최소 단어 수 (기본값: 70)
- **`max_phrase_length`**: 반복 구절을 감지하기 위한 구절 내 최대 단어 수 (기본값: 2)

## 사용 예시

### 학습
`main_config.yaml`을 사용하여 학습을 실행:

```bash
./run.sh
```

### 평가
`eval_config.yaml`을 사용하여 모델 성능 평가:

```bash
./evaluate.sh
```

### 추론
`inference_config.yaml`을 사용하여 문장 분리 추론:

```bash
./inference.sh
```

> **참고**: 각 설정 파일을 수정하여 자신의 데이터 및 환경에 맞게 최적화할 수 있습니다.