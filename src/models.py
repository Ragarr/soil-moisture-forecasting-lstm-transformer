# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import os

# -----------------------
# 1) Hiperparámetros default
# -----------------------
BATCH_SIZE      = 16        # batch size pequeño para dataset reducido
DROPOUT_RATE    = 0.3       # dropout para LSTM y Transformer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.double)
print(f"Using device: {device}")

#########################
# Positional Encoding for Transformer
#########################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
         super(PositionalEncoding, self).__init__()
         self.dropout = nn.Dropout(p=dropout)
         pe = torch.zeros(max_len, d_model)
         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
         pe[:, 0::2] = torch.sin(position * div_term)
         pe[:, 1::2] = torch.cos(position * div_term)
         pe = pe.unsqueeze(0)  # (1, max_len, d_model)
         self.register_buffer('pe', pe)
    
    def forward(self, x):
         x = x + self.pe[:, :x.size(1)]
         return self.dropout(x)

# -----------------------
# 2) Modelos con dropout
# -----------------------
class LSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=16, num_layers=2, dropout=DROPOUT_RATE):
        super().__init__()
        # dropout interno entre capas LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, horizon=1):
        self.lstm.flatten_parameters()
        batch_size, lookback, _ = x.size()
        preds = []
        current_input = x.clone()
        for _ in range(horizon):
            out, _ = self.lstm(current_input)
            last = out[:, -1, :]
            last = self.dropout(last)
            pred = self.fc(last)
            preds.append(pred.unsqueeze(1))
            new_exog = current_input[:, -1, :6]  # Solo las primeras 6 features
            new_step = torch.cat([new_exog, pred], dim=1).unsqueeze(1)
            current_input = torch.cat([current_input[:, 1:, :], new_step], dim=1)
        return torch.cat(preds, dim=1)

class Transformer(nn.Module):
    def __init__(self, d_model=8, nhead=1, d_ff=10, num_layers=1,
                 input_size=6, dropout=DROPOUT_RATE):
        super().__init__()
        self.fc_in = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x, horizon=1):
        batch_size, lookback, _ = x.size()
        preds = []
        current = x.clone()
        for _ in range(horizon):
            enc = self.fc_in(current)
            enc = self.pos_encoder(enc)
            enc = enc.transpose(0,1)
            enc = self.transformer_encoder(enc)
            enc = enc.transpose(0,1)
            last = enc[:, -1, :]
            pred = self.fc_out(last)
            preds.append(pred.unsqueeze(1))
            new_exog = current[:, -1, :6]  # Solo las primeras 6 features
            new_step = torch.cat([new_exog, pred], dim=1).unsqueeze(1)
            current = torch.cat([current[:,1:,:], new_step], dim=1)
        return torch.cat(preds, dim=1)


#########################
# Modelo Dummy para predicción basada en la media
#########################
class DummyModel(nn.Module):
    def __init__(self):
         super(DummyModel, self).__init__()
    def forward(self, x, horizon=1):
         # Se asume que la última columna de x es la variable a predecir
         # Se calcula la media a lo largo del eje temporal (lookback) para cada muestra
         avg = x[:, :, -1].mean(dim=1)  # (batch,)
         # Se repite el valor para cubrir todos los pasos del horizonte
         pred = avg.unsqueeze(1).repeat(1, horizon).unsqueeze(2)  # (batch, horizon, 1)
         return pred


#########################
# Data loading and preprocessing
#########################
def load_and_preprocess_data(data_path, station_name, depth):
    # Construir la ruta completa al archivo usando os.path.join
    filepath = os.path.join(data_path, station_name + ".csv")
    data = pd.read_csv(filepath)
    # si hay mas de 20 filas con algun nan, eliminar todas las filas posteriores a la primera

    if data.isnull().sum().sum() > 20:
        first_nan = data.isnull().any(axis=1).idxmax()
        data = data.iloc[:first_nan]

    target_col = 'soil_moisture'
    theta_star = data[target_col].values[:, None] * 100
    theta_star = np.where(theta_star > -998, theta_star, math.nan)

    max_val, min_val = np.nanmax(theta_star), np.nanmin(theta_star)
    theta_norm = 2 * (theta_star - min_val) / (max_val - min_val) - 1

    features = ['precipitacion', 'temperatura', 'humedad_ambiente', 'viento', 'radiacion_solar',  'soil_moisture']


    inputs = data[features].copy().iloc[1:]
    for f in features:
        inputs[f] = np.where(inputs[f] > -998, inputs[f], math.nan)
    max_feat, min_feat = np.nanmax(inputs.to_numpy()), np.nanmin(inputs.to_numpy())
    inputs = 2 * (inputs - min_feat) / (max_feat - min_feat) - 1
    
    data_series = np.hstack((inputs.to_numpy(), theta_norm[1:]))
    targets = theta_norm[1:]
    return data_series, targets, max_val, min_val

#########################
# Batch generator
#########################
def generate_sequences(x, y, lookback, horizon):
    seq_x, seq_y = [], []
    for i in range(len(x) - lookback - horizon + 1):
        if not np.isnan(np.sum(x[i:i + lookback])) and not np.isnan(np.sum(y[i + lookback:i + lookback + horizon])):
            seq_x.append(x[i:i + lookback])
            seq_y.append(y[i + lookback:i + lookback + horizon])
    seq_x = np.array(seq_x)
    seq_y = np.array(seq_y)
    return torch.from_numpy(seq_x).to(device), torch.from_numpy(seq_y).to(device)
