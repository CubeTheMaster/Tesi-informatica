import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CICDDoSCombinedDataset(Dataset):
    def __init__(self, file_paths, window_size=16):
        self.window_size = window_size
        
        all_df = []
        print(f"Caricamento di {len(file_paths)} file in memoria...")
        
        for f in file_paths:
            print(f"Leggendo: {f}")
            df = pd.read_parquet(f, engine='pyarrow')
            # Pulizia immediata per risparmiare RAM
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            all_df.append(df)
        
        # Unione di tutti i file
        full_df = pd.concat(all_df, ignore_index=True)
        del all_df # Libera la lista originale
        
        # 1. Encoding Etichette
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(full_df['Label'])
        
        # 2. Selezione Feature
        cols_to_drop = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 
                        'Destination IP', 'Timestamp', 'Label', 'SimillarHTTP']
        X_raw = full_df.drop(columns=[c for c in cols_to_drop if c in full_df.columns], errors='ignore')
        
        # 3. Normalizzazione
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X_raw)
        
        # Conversione in tensori
        self.y = torch.tensor(y, dtype=torch.long)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        
        print(f"Dataset pronto! Totale righe: {len(self.X)} | Classi: {self.label_encoder.classes_}")

    def __len__(self):
        return len(self.X) - self.window_size + 1

    def __getitem__(self, idx):
        return self.X[idx : idx + self.window_size], self.y[idx + self.window_size - 1]