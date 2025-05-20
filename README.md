# ESC: Entropy-guided Semantic Coupling for Robust Cross-modal Retrieval

This project implements the **ESC (Entropy-guided Semantic Coupling)** framework for robust cross-modal (text-image) retrieval, as described in our paper. ESC addresses the instability and gradient divergence in unconstrained multimodal contrastive learning by introducing:

- **Multi-Level Entropy Control (MLEC):** Dynamically quantifies semantic complexity within and across modalities, providing entropy signals for adaptive optimization.
- **Parameter-Level Bidirectional Coupling (PLBC):** Uses entropy signals to adaptively couple gradient updates between modalities, aligning optimization trajectories and mitigating conflicts.

## Features
- Large-scale cross-modal retrieval (e.g., Flickr30k, ViQuAE)
- Entropy-driven, adaptive gradient coupling for robust alignment
- End-to-end training with a composite loss: InfoNCE, Jensen-Shannon Divergence (JSD), and consistency regularization
- PyTorch-based, modular and extensible

## Installation

```bash
pip install -r requirements.txt
```

## Method Overview

**ESC** consists of two main modules:
- **MLEC:** Computes intra-modal and cross-modal entropy using Gaussian kernels and similarity matrices, capturing semantic uncertainty.
- **PLBC:** Dynamically adjusts gradient coupling coefficients based on entropy differences, ensuring stable and adaptive cross-modal optimization.
- **Total Loss:** Combines InfoNCE, JSD, and a consistency loss (1-cosine similarity), with a scaling factor γ for the consistency term.

For detailed methodology, please refer to the Methodology section of our paper.

## Requirements
- Python >= 3.8
- torch >= 2.0.0
- transformers >= 4.30.0
- See `requirements.txt` for full list

## Project Structure
```
.
├── models/           # ESC core implementation (MLEC, PLBC, etc.)
├── encoder/        
├── scripts/        
├── data/           
├── requirements.txt
└── README.md
```


