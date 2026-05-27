import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
import glob
import shutil
import kagglehub
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt              
import numpy as np                               
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns                            

from mamba3 import Mamba3Config, get_device
from Dataset_CiCDDoS import CICDDoSCombinedDataset
from Model import Mamba3MultiClass

def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    return f"{int(mins)}m {int(secs)}s"

# --- FUNZIONE DI DOWNLOAD ---
def download_full_dataset(target_folder):
    os.makedirs(target_folder, exist_ok=True)
    existing_files = glob.glob(os.path.join(target_folder, "*.parquet"))
    if existing_files:
        print(f"[DATASET] Rilevati {len(existing_files)} file Parquet. Salto il download.")
        return

    print("[DATASET] Inizio download completo da Kaggle...")
    try:
        cache_path = kagglehub.dataset_download("dhoogla/cicddos2019")
        copied_count = 0
        for root, dirs, files in os.walk(cache_path):
            for file_name in files:
                if file_name.endswith(".parquet"):
                    shutil.copy(os.path.join(root, file_name), os.path.join(target_folder, file_name))
                    copied_count += 1
        print(f"[DATASET] Download e trasferimento completati: {copied_count} file pronti.")
    except Exception as e:
        print(f"[ERRORE] Errore nel download: {e}")

# --- FUNZIONE DI SALVATAGGIO GRAFICI E MATRICI ---
def save_final_artifacts(all_preds, all_targets, class_names, output_dir, epoch, metrics, train_losses):
    num_classes = len(class_names)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    
    # 1. MATRICE ASSOLUTA (COUNTS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix (Counts) - Epoca: {epoch} (F1-Macro: {metrics['f1_macro']:.4f})")
    plt.ylabel("Classe Reale")
    plt.xlabel("Classe Predetta")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    file_counts = os.path.join(output_dir, "best_confusion_matrix_counts.png")
    plt.savefig(file_counts, dpi=300)
    plt.close()
    
    # 2. MATRICE NORMALIZZATA (PERCENTAGES)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix (Normalized) - Epoca: {epoch} (F1-Macro: {metrics['f1_macro']:.4f})")
    plt.ylabel("Classe Reale")
    plt.xlabel("Classe Predetta")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    file_norm = os.path.join(output_dir, "best_confusion_matrix_normalized.png")
    plt.savefig(file_norm, dpi=300)
    plt.close()
    print(f"[GRAFICI] Matrici di confusione salvate con successo.")

    # 3. GRAFICO CURVA DELLA LOSS
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', color='crimson', linewidth=2)
    plt.title("Curva di Apprendimento del Modello (Training Loss)")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(range(1, len(train_losses) + 1))
    plt.tight_layout()
    file_loss = os.path.join(output_dir, "learning_loss_curve.png")
    plt.savefig(file_loss, dpi=300)
    plt.close()
    print(f"[GRAFICO] Curva della Loss salvata in: {file_loss}")

def evaluate_for_metrics(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    # Calcolo analitico delle metriche avanzate
    acc = (100 * correct / total) if total > 0 else 0
    prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    metrics = {
        'accuracy': acc,
        'precision_macro': prec,
        'recall_macro': rec,
        'f1_macro': f1
    }
    return metrics, all_preds, all_targets

def train():
    device = get_device()
    start_time_total = time.time()
    data_folder = "/dataset" 
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    session_output_dir = os.path.join(".", "output", f"run_{timestamp}")
    os.makedirs(session_output_dir, exist_ok=True)

    download_full_dataset(data_folder)

    all_files = sorted(glob.glob(os.path.join(data_folder, "*.parquet")))
    if not all_files:
        print("[ERRORE] Nessun file Parquet presente nella cartella!")
        return

    print(f"[SISTEMA] Trovati {len(all_files)} file parquet totali. Inizio unificazione...")
    list_dfs = []
    allowed_classes = ['Benign', 'LDAP', 'MSSQL', 'NetBIOS', 'UDP', 'UDP-Lag', 'Syn']

    for f in all_files:
        df_temp = pd.read_parquet(f, engine='pyarrow')
        df_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_temp.dropna(inplace=True)
        df_temp = df_temp[df_temp['Label'].isin(allowed_classes)]
        if not df_temp.empty:
            list_dfs.append(df_temp)

    full_df = pd.concat(list_dfs, ignore_index=True)
    del list_dfs 
    print(f"[SISTEMA] Dataset globale unificato. Righe totali: {len(full_df)}")

    le = LabelEncoder()
    le.fit(allowed_classes)

    print("[SISTEMA] Applicazione dello Split Stratificato (80/20)...")
    train_df, test_df = train_test_split(
        full_df, 
        test_size=0.20, 
        random_state=42, 
        stratify=full_df['Label']
    )
    del full_df 

    train_dataset = CICDDoSCombinedDataset(train_df, le, window_size=16)
    test_dataset = CICDDoSCombinedDataset(test_df, le, window_size=16)
    del train_df, test_df 

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    actual_num_classes = len(le.classes_)
    args = Mamba3Config(d_model=32, n_layer=1, d_state=16)
    model = Mamba3MultiClass(args, input_dim=77, num_classes=actual_num_classes).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Liste e variabili per lo storico e la selezione dell'epoca ottimale
    train_losses = []
    best_f1 = -1.0
    best_epoch = -1
    best_predictions = None
    best_targets = None
    best_metrics = None

    print(f"\nInizio Addestramento Avanzato su: {device}")
    epochs = 5
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        
        running_loss = 0.0
        steps = 0
        progress_bar = tqdm(train_loader, desc=f"Epoca {epoch}")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            if torch.isnan(loss): 
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            steps += 1
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / steps if steps > 0 else 0.0
        train_losses.append(epoch_loss)

        # Valutazione multiclasse estesa a fine epoca
        metrics, preds, targets = evaluate_for_metrics(model, test_loader, device)
        print(f"Risultati Epoca {epoch}:")
        print(f"  -> Accuracy:  {metrics['accuracy']:.2f}%")
        print(f"  -> Precision: {metrics['precision_macro']:.4f}")
        print(f"  -> Recall:    {metrics['recall_macro']:.4f}")
        print(f"  -> F1-Macro:  {metrics['f1_macro']:.4f}")
        
        # Scegliamo la migliore epoca basandoci sulla F1-Macro (metrica accademica ideale)
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            best_epoch = epoch
            best_predictions = preds
            best_targets = targets
            best_metrics = metrics
        
        print(f"Tempo Epoca: {format_time(time.time() - epoch_start)}\n")

    # --- SALVATAGGIO DEI REPORT E DEI GRAFICI FINALI ---
    print(f"\n--- Addestramento Terminato ---")
    if best_predictions is not None:
        print(f"\n[PROCESSO] Esportazione report definitivo basato sull'Epoca {best_epoch}:")
        print(f"Miglior Accuracy:  {best_metrics['accuracy']:.2f}%")
        print(f"Miglior Precision: {best_metrics['precision_macro']:.4f}")
        print(f"Miglior Recall:    {best_metrics['recall_macro']:.4f}")
        print(f"Miglior F1-Macro:  {best_metrics['f1_macro']:.4f}")
        
        save_final_artifacts(best_predictions, best_targets, allowed_classes, session_output_dir, best_epoch, best_metrics, train_losses)
    
    print(f"Durata totale elaborazione: {format_time(time.time() - start_time_total)}")

if __name__ == "__main__":
    train()