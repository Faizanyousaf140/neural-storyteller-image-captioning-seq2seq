# Image Captioning with Deep Learning — Gen AI Assignment 1

## Project Overview
This project implements an **end-to-end image captioning system** using deep learning:
- **Encoder**: Pre-trained ResNet-50 (CNN) for extracting visual features
- **Decoder**: LSTM (RNN) for generating captions word by word
- **Dataset**: Flickr30k — 31,783 images with 5 captions each

## Project Structure
```
gen-ai-assignment-1/
├── app.py                      # Streamlit web app for interactive caption generation
├── model/
│   └── best_model.pth          # Trained model checkpoint (place here!)
├── requirements.txt            # Python dependencies
├── Assignment1_Generative_AI_22F-3875.ipynb  
└── README.md                   # This file
```

## Setup Instructions

### 1. Place the Model File
Download `best_model.pth` from your Kaggle notebook output and place it in the `model/` directory:
```
model/best_model.pth
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

### 4. Open in VS Code
Open the project folder in VS Code:
```bash
code .
```

## Model Architecture
| Component        | Details                          |
|-----------------|----------------------------------|
| Encoder         | ResNet-50 → Linear(2048, 512) → ReLU |
| Decoder         | Embedding(vocab, 256) → LSTM(256, 512) → Linear(512, vocab) |
| Total Parameters | ~8.57 million                    |
| Training Epochs | 15 (with early stopping)         |
| Best Val Loss   | 3.0251                           |

## Evaluation Results
| Method          | BLEU-4 | Precision | Recall | F1-Score |
|----------------|--------|-----------|--------|----------|
| Greedy Search  | 0.1493 | 0.5608    | 0.1578 | 0.2463   |
| Beam Search (k=3) | 0.1648 | 0.5859 | 0.1546 | 0.2446   |

Beam Search outperforms Greedy Search by approximately **10.4%** on BLEU-4.

## Usage in VS Code
1. Open the project folder in VS Code
2. The model file at `model/best_model.pth` can be loaded directly using:
```python
import torch
checkpoint = torch.load("model/best_model.pth", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
vocab = checkpoint['vocab']
```
> **Note**: The model file (`best_model.pth`) is stored using [Git LFS](https://git-lfs.com/). 
> To clone this repo properly, install Git LFS first:
> ```bash
> git lfs install
> git clone https://github.com/Ahmad-211/Neural-Storyteller-Image-Captioning-with-Seq2Seq.git
> ```

## Authors
 - Muhammad Faizan Yousaf
