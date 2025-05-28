# Spoof or Proof: Generalizable Deepfake Audio Detection

This repository contains the code, models, and preprocessing scripts used in the paper:  
**"Beyond Speaker Identity: Generalizable Deepfake Detection using Audio Embeddings and Time Delay Neural Networks"**  
Author: Devon Campbell 

## Overview

This project develops and evaluates a robust audio deepfake detection system that generalizes to speakers and spoofing techniques unseen during training. It uses x-vector audio embeddings and evaluates similarity metrics (cosine, Euclidean, Minkowski, Manhattan) between test and reference samples.

Two architectures are included:
- **TDNN-based embedding extractor** (using Kaldi-generated x-vectors)
- **Transformer-based classifier** for supervised detection

## Project Structure

```bash
├── cbms.py                # Core logic for centroid and max similarity analysis
├── transformer.py         # Transformer-based classifier and training routines
├── utils.py               # Dataset and preprocessing utilities
├── asv2xvector.sh         # Bash script for generating x-vectors using Kaldi
├── report.pdf             # Full technical report describing methodology and experiments
```

## Installation

Dependencies:
- Python 3.8+
- PyTorch
- torchaudio
- kaldiio
- librosa
- pandas, numpy, matplotlib
- Kaldi (for running `asv2xvector.sh`)

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

> ⚠️ Kaldi must be installed separately and the environment properly configured to run x-vector extraction.

## Usage

### 1. Generate X-vectors

Run the Kaldi pipeline to extract x-vectors from ASVspoof and VoxCeleb datasets:

```bash
bash asv2xvector.sh
```

This script:

* Converts `.flac` and `.wav` files to `wav.scp`
* Computes MFCCs and VAD
* Extracts x-vectors from pretrained Kaldi model

### 2. Run Centroid-Based Similarity Detection

```bash
python cbms.py
```

This will:

* Load train/dev/eval x-vectors
* Compute centroid of reference embeddings
* Evaluate test samples using similarity thresholds
* Output accuracy, F1 score, and DET curves per metric

### 3. Train and Evaluate Transformer Classifier

Edit the `main()` method in `transformer.py` to configure training:

```bash
python transformer.py
```

Supports:

* Pretraining on VoxCeleb embeddings
* Fine-tuning and evaluation on ASVspoof
* Outputs DET curves and EER

## Results

The cosine similarity metric outperformed others in detecting deepfake audio, achieving:

* **Accuracy**: 78.91% on evaluation set
* **F1 Score**: 40.65%
* **EER**: 26.51%

See `report.pdf` for a complete analysis.
