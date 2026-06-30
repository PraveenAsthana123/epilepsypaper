# Advanced Models — CNN / RNN / Transformer / RL

PyTorch (CPU), UCI balanced subsample (4600 segments), epoch-level 80/20. Real data, no fabrication (Sec. 57.7).

| Model | Type | Acc | Sens | Spec | F1 | AUC |
|---|---|---|---|---|---|---|
| CNN_computer_vision | Computer Vision (CNN) | 0.9489 | 0.9304 | 0.9674 | 0.948 | 0.9914 |
| TimeSeries_TCN | Time-Series (1D temporal CNN) | 0.9489 | 0.9413 | 0.9565 | 0.9485 | 0.9861 |
| RNN_LSTM | RNN (LSTM) | 0.95 | 0.9435 | 0.9565 | 0.9497 | 0.9792 |
| Transformer | Transformer (attention) | 0.95 | 0.9478 | 0.9522 | 0.9499 | 0.9904 |
| RL_REINFORCE | Reinforcement Learning | 0.963 | 0.9522 | 0.9739 | 0.9626 | 0.9918 |
