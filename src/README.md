```
.
├── KSSDS
│   ├── T5_encoder.py   # T5 모델을 커스터마이징한 Encoder 모듈
│   ├── __init__.py
│   ├── dataloader.py   # 커스텀 데이터로더
│   ├── dataset.py      # 커스텀 데이터셋
│   ├── evaluate.py     # 평가 스크립트
│   ├── inference.py    # 추론 스크립트
│   ├── main.py         # 학습과 평가의 엔트리 포인트
│   ├── train.py        # 학습 파이프라인 스크립트
│   ├── trainer.py      # 커스텀 트레이너
│   └── util.py         # 공통적으로 사용하는 유틸리티 함수들 (예: 텐서보드 콜백 함수 등)

```