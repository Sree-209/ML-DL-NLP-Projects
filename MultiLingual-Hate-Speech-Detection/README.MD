# Multilingual Hate Speech Detection

This project tackles the problem of detecting hate speech in multilingual and code-mixed content, including both text and memes. It combines deep learning with handcrafted features to improve classification performance on noisy real-world data.

## Problem Statement

Online platforms are often flooded with toxic and offensive content, especially in multilingual or code-mixed formats (e.g., Hindi-English). This project aims to detect such hate speech in both user comments and memes, and optionally classify its severity.

## Approach

**Text-Based Detection:**
- Preprocessing: removal of emojis, special characters, and language normalization
- Feature Engineering: Hinglish profanity score, binary hate flag, normalized text length
- Embeddings: XLM-RoBERTa, SBERT
- Architecture: Attention-based BiGRU + Transformer + MLP
- Loss Function: Focal Loss to address class imbalance

**Meme-Based Detection:**
- Text extraction using PaddleOCR
- Hate speech analysis of extracted text using the same pipeline

## Datasets Used

- **[Hate Speech in Memes (HSIM)](https://aclanthology.org/2024.lrec-main.300.pdf)** – Detecting hate speech from memes using OCR-extracted text.
- **[Hindi-English Code-Mixed Twitter Dataset](https://data.mendeley.com/datasets/snc7mxpj6t/1)** – Tweets labeled for hate speech detection in code-mixed Hinglish.
- **[Hinglish Offensive Text Classification](https://github.com/pmathur5k10/Hinglish-Offensive-Text-Classification)** – Contains Hinglish comments labeled for offensive language, used to validate and augment profanity detection.

> Note: Raw images and large datasets are excluded from this repository. Files are referenced but not committed to maintain size efficiency.

## Goals

- Detect hate speech in multilingual and code-mixed comments and memes
- Improve classification accuracy and recall in imbalanced datasets
- Experiment with severity tagging (low, moderate, high)

## Tools and Libraries

- Python, PyTorch
- HuggingFace Transformers (XLM-RoBERTa)
- PaddleOCR for meme analysis
- SentenceTransformers
- Scikit-learn, Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

## Exclusions

- The `dataset/` directory and large `.csv` files have been excluded from version control via `.gitignore`.
- The repository contains only code and essential configuration for reproducibility.

## Status

Model architecture and feature fusion are complete. Ongoing efforts are focused on fine-tuning and maximizing recall and F1-score on hate speech classification.
