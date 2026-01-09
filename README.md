
# Knowledge Distillation for Commodity Forecasting

Reusable Python project for uncertainty-aware multi-teacher
knowledge distillation in single-commodity time-series forecasting.

## Features
- Input: CSV with {date, price}
- Horizon: 6 months
- Teachers: DLinear, PatchTST
- Student: MLP
- Uncertainty-aware KD
- Google Colab friendly

## Run
1. Upload folder to Colab
2. Place dataset in data/
3. Edit DATA_PATH in run.py
4. Run:
   !python run.py
