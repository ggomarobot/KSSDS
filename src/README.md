```
.
├── KSSDS/              # PyPI 패키지화에 필요한 핵심 코드
│   ├── __init__.py
│   ├── inference.py    # KSSDS 패키지에서 추론 제공
│   └── T5_encoder.py   # T5 모델을 커스터마이징한 Encoder 모듈
│ 
└── utils/              # 학습 및 평가용 유틸리티
    ├── dataloader.py   # 커스텀 데이터로더
    ├── dataset.py      # 커스텀 데이터셋
    ├── trainer.py      # 커스텀 트레이너
    └── util.py         # 공통 유틸리티 (예: 텐서보드 콜백 함수)
    
```