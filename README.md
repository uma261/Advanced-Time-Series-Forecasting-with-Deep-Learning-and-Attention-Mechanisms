# Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms

This project implements a production-quality PyTorch pipeline for multivariate time-series forecasting using an LSTM augmented with a self-attention mechanism.

## Features
- Synthetic multivariate dataset generator with trend + two seasonalities + noise.
- PyTorch Dataset and DataLoader for sequence-to-one forecasting.
- LSTM encoder + Attention decoder implementation.
- Training loop with checkpointing.
- Evaluation (RMSE, MAE) and baseline (vanilla LSTM) comparison.
- Reproducible (fixed seeds).

## Files
- `main.py` - end-to-end script: data generation, training, evaluation.
- `model.py` - model classes: LSTMBaseline and LSTMAttention.
- `utils.py` - helper functions: metrics, dataset class, seeding.
- `requirements.txt` - Python packages required.

## Quickstart
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --epochs 30 --batch_size 64
```

The script saves `best_attention.pt` and `best_lstm.pt` model checkpoints and prints final RMSE/MAE for test set.
