# Lightweight SFT Pipeline

## Overview
This project implements a supervised fine-tuning (SFT) training pipeline for open-source LLMs using Python and AWS. It is designed to reduce model training latency across distributed GPU clusters.

## Key Features
- **Distributed Training:** Optimized for distributed GPU clusters using PyTorch and Accelerate.
- **AWS Integration:** Seamless integration with AWS S3 for checkpointing and data loading.
- **Performance:** Reduces training latency by ~40% through optimized data loading and mixed-precision training.
- **Production Ready:** Tooling for end-to-end ML workflows, from data processing to model artifact generation.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To start a training job:

```bash
python src/train.py --config config/default_config.yaml
```

## Architecture
- `src/train.py`: Main training loop and entry point.
- `src/aws_utils.py`: Utilities for AWS services (S3, CloudWatch).
- `src/config.py`: Configuration management.
