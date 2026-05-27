import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CICDDoSCombinedDataset(Dataset):
    def __init__(self, df, label_encoder, window_size=16):
        self.window_size = window_size
        self.label_encoder = label_encoder
        
        # Le etichette numeriche (indici da 0 a 6)
        y = self.label_encoder.transform(df['Label'])
        
        # Selezione delle Feature numeriche utili eliminando i metadati
        cols_to_drop = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 
                        'Destination IP', 'Timestamp', 'Label', 'SimillarHTTP']
        X_raw = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        # Normalizzazione dei dati Z-score
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X_raw)
        
        # Conversione finale nei tensori di PyTorch richiesti dal Mamba-3
        self.y = torch.tensor(y, dtype=torch.long)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        
        print(f"[DATASET] Pronto! Campioni totali: {len(self.X)} | Classi: {list(self.label_encoder.classes_)}")

    def __len__(self):
        return len(self.X) - self.window_size + 1

    def __getitem__(self, idx):
        return self.X[idx : idx + self.window_size], self.y[idx + self.window_size - 1]