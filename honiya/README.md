# Regularized Logistic Regression & Factor Mining Pipeline

## Overview
This workstream implements a regularized linear classification pipeline designed to score and rank CVEs for the CISA KEV benchmark while strictly avoiding post-breach data leakage.

## Folder Map
- `scripts/`: Self-contained training and inference pipeline (`train_and_predict.py`).
- `submissions/`: Output `top500_regularized_lr_capped.csv` containing the final 500 ranked CVE IDs.
- `charts/`: Visualizations of benchmark recall and factor discovery analysis.
- `context_honiya.md`: Full technical handoff and evaluation report.

## How to Run
From the root of the repository:
python honiya/scripts/train_and_predict.py

Reads raw data from: `IBM-20261107T174321Z-1-001/IBM/data/data/`