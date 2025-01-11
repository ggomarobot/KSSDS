## Data 폴더 구조
```
data
├── 1. raw                               # AI HUB 원본 데이터
│   ├── 1. train                        
│   │   ├── audio                        # Whisper를 통해 transcribe된 AI HUB 음성 데이터
│   │   └── text                         # AI HUB에서 제공된 문장 단위 텍스트 데이터
│   │
│   ├── 2. validation                  
│   │   └── audio                        # Whisper를 통해 transcribe된 AI HUB 음성 데이터
│   │
│   ├── 3. test                         
│   │   ├── audio                        # Whisper를 통해 transcribe된 AI HUB 음성 데이터
│   │   └── text                         # AI HUB에서 제공된 문장 단위 텍스트 데이터
│   │
│   └── script                          
│       ├── step1.sh                    
│       ├── step1_config.yaml           
│       └── whisper_process_for_audio.py # AI HUB에서 받은 음성 파일들을 Whisper를 활용해 transcribe하는 코드
│
├── 2. LLM split                         # LLM(ChatGPT, Gemini)을 사용해 분리된 데이터
│   ├── 1. train
│   │   └── audio
│   │       ├── ChatGPT                  # ChatGPT를 사용해 문장 단위로 분리된 Whisper transcription
│   │       └── Gemini                   # Gemini를 사용해 문장 단위로 분리된 Whisper transcription
│   │
│   ├── 2. validation
│   │   └── audio
│   │       ├── ChatGPT                
│   │       └── Gemini                 
│   │
│   ├── 3. test
│   │   └── audio
│   │       ├── ChatGPT                
│   │       └── Gemini                 
│   │
│   └── script
│       └── LLM_split.py                # LLM을 활용해 "1. raw"의 음성 데이터를 분리하는 코드
│
├── 3. confidence evaluation            # LLM 결과에 대한 신뢰도 평가
│   ├── 1. train
│   │   └── audio                       # 신뢰도 평가 후 생성된 학습 데이터
│   │
│   └── script
│       ├── confidence_evaluation.py    # "2. LLM split"의 문장 데이터를 4단계 필터링을 통해 선별하는 코드 (잘린 문장 품질 관리)
│       ├── step3.sh                    
│       └── step3_config.yaml           
│
├── 4. labeled                          # 종결 어미 라벨링 완료 데이터
│   ├── 1. train
│   │   ├── audio
│   │   └── text
│   │
│   ├── 2. validation
│   │   └── audio
│   │       ├── ChatGPT                
│   │       └── Gemini                 
│   │
│   ├── 3. test
│   │   ├── audio
│   │   └── text
│   │
│   └── script
│       ├── label.py                    # 라벨링 코드
│       ├── step4.sh                    
│       └── step4_config.yaml           
│
├── 5. final                            # 최종 데이터셋
│   ├── 1. train
│   │   ├── audio
│   │   └── text
│   │
│   ├── 2. validation
│   │   ├── ChatGPT
│   │   └── Gemini
│   │
│   ├── 3. test
│   │   ├── audio
│   │   └── text
│   │
│   └── script
│       ├── concatenate_sentences.py    # 개별 문장을 모델 학습을 위해 Long-form Text로 연결하는 스크립트
│       ├── remove_space.py             # concatenate_sentences.py 실행 후 생성된 문장 맨 앞 공백 제거 스크립트
│       ├── step5.sh                    
│       └── step5_config.yaml           
│
├── 6. no length filter (ablation study)# "3. confidence evaluation"을 건너뛰고 생성한 데이터 (Ablation Study 용)
│
└── misc                                
    └── sanity_check_and_split.py       # sanity check: 최종 생성된 Long-form Text 의 label 갯수와 token 갯수 일치 여부 확인
                                        # split       : 종결 어미로 labeling 된 곳을 기준으로 잘라 문장 단위의 원본 데이터로 돌아오는지 확인
```

### 주요 폴더 설명

- **1. raw**  
  AI HUB에서 제공받은 원본 데이터로 구성됩니다.  
  - `audio`: Whisper를 사용해 음성 데이터를 transcribe한 결과.
  - `text`: AI HUB에서 제공된 문장 단위 텍스트 데이터.
  - `script`: Whisper를 활용한 transcription 코드를 포함합니다.

- **2. LLM split**  
  LLM(ChatGPT, Gemini)을 활용하여 "1. raw"의 audio 데이터를 문장 단위로 분리한 데이터셋입니다.  
  - `audio`: ChatGPT 및 Gemini로 분리된 데이터.
  - `script`: LLM 기반 문장 분리 스크립트를 포함합니다.

- **3. confidence evaluation**  
  LLM으로 분리된 데이터의 품질을 평가하고, 신뢰도가 높은 데이터를 선별합니다.  
  - `audio`: 신뢰도 평가를 거친 학습 데이터.
  - `script`: LLM 결과를 필터링하는 스크립트를 포함합니다.

- **4. labeled**  
  종결 어미 라벨링이 완료된 데이터셋입니다.

- **5. final**  
  최종적으로 학습에 사용될 데이터셋으로, Long-form Text로 결합되어 모델 학습 준비가 완료된 상태입니다.  
  - `concatenate_sentences.py`: 개별 문장을 Long-form Text로 연결.
  - `remove_space.py`: 연결 과정에서 생긴 불필요한 공백 제거 스크립트.

- **6. no length filter (ablation study)**  
  "3. confidence evaluation" 단계를 생략한 데이터셋으로, ablation study를 위해 생성되었습니다.

- **misc**  
  - `sanity_check_and_split.py`: 최종 데이터의 라벨과 토큰 갯수가 일치하는지 검증 및 디버깅 용으로 다시 개별 문장으로 되돌리기 위한 스크립트.

---

### Google Drive 링크
`data` 폴더와 관련된 전체 데이터셋은 [Google Drive 링크](https://drive.google.com/drive/folders/19qa1AKetvRZvfOWcGL_BOj-z43sTElhH?usp=sharing)를 통해 확인할 수 있습니다. 
각 폴더에 대한 자세한 내용은 위 설명을 참고해주세요.