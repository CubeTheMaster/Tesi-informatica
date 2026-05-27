import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

class UNSWCombinedDataset(Dataset):
    def __init__(self, df, target_encoder, window_size=16):
        self.window_size = window_size
        self.target_encoder = target_encoder
        
        # Mappatura delle categorie di attacco (Label numeriche)
        y = self.target_encoder.transform(df['attack_cat'])

        # Rimozione dei metadati, delle etichette di testo e del flag binario 'label'
        cols_to_drop = ['id', 'attack_cat', 'label']
        X_raw = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        # Gestione automatica ed encoding delle colonne categoriche presenti in UNSW
        categorical_cols = ['proto', 'service', 'state']
        for col in categorical_cols:
            if col in X_raw.columns:
                fe = LabelEncoder()
                X_raw[col] = fe.fit_transform(X_raw[col].astype(str))

        # Normalizzazione Z-score standard delle feature numeriche
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X_raw)
            
        # Conversione finale nei tensori richiesti dal Mamba-3
        self.y = torch.tensor(y, dtype=torch.long)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        
        print(f"[DATASET UNSW] Pronto! Campioni totali: {len(self.X)} | Classi: {list(self.target_encoder.classes_)}")

    def __len__(self):
        return len(self.X) - self.window_size + 1

    def __getitem__(self, idx):
        return self.X[idx : idx + self.window_size], self.y[idx + self.window_size - 1]