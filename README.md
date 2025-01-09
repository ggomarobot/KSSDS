# KSSDS: Korean Sentence Splitter for Dialogue Systems

[![PyPI version](https://badge.fury.io/py/KSSDS.svg)](https://pypi.org/project/KSSDS/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

KSSDS is a robust Korean sentence splitter designed specifically for dialogue systems. It handles complex STT hallucinations, abnormal repetitions, and maintains semantic integrity, making it suitable for tasks such as machine translation, sentiment analysis, and more.

---

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Folder Structure](#folder-structure)
5. [License](#license)
6. [Acknowledgments](#acknowledgments)

---

## Installation

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
