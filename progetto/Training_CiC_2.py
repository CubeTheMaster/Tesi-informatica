import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
import glob
import shutil    
import kagglehub  
from tqdm import tqdm
from mamba3 import Mamba3Config, get_device
from Dataset_CiCDDoS import CICDDoSCombinedDataset
from Model import Mamba3MultiClass

def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    return f"{int(mins)}m {int(secs)}s"

# --- FUNZIONE DI DOWNLOAD AUTOMATICO E MIRATO ---
def download_dataset_via_kaggle(target_folder):
    """Scarica il dataset da Kaggle e posiziona SOLO i file LDAP e MSSQL specificati nella cartella target"""
    os.makedirs(target_folder, exist_ok=True)
    
    # I 4 file esatti richiesti
    target_files = [
        "LDAP-testing.parquet",
        "LDAP-training.parquet",
        "MSSQL-testing.parquet",
        "MSSQL-training.parquet"
    ]
    
    # Controlliamo se sono già tutti presenti per evitare di ripetere il processo
    missing_files = [f for f in target_files if not os.path.exists(os.path.join(target_folder, f))]
    
    if not missing_files:
        print(f"[DATASET] Tutti i file richiesti (LDAP e MSSQL) sono già presenti in {target_folder}. Salto il download.")
        return

    print(f"\n[DATASET] File mancanti rilevati: {missing_files}")
    print("[DATASET] Inizio download automatico da Kaggle tramite kagglehub...")
    
    try:
        # Scarica il dataset (kagglehub sincronizza l'archivio nella cache locale)
        cache_path = kagglehub.dataset_download("dhoogla/cicddos2019")
        print(f"[DATASET] Dataset sincronizzato in cache: {cache_path}")
        
        print(f"[DATASET] Filtro e copia dei file mirati verso {target_folder}...")
        copied_count = 0
        
        # Scansioniamo la cache alla ricerca solo dei 4 file che ci interessano
        for root, dirs, files in os.walk(cache_path):
            for file_name in files:
                if file_name in target_files:
                    file_path = os.path.join(root, file_name)
                    dest_path = os.path.join(target_folder, file_name)
                    
                    # Copia il file solo se non esiste già nella cartella di destinazione
                    if not os.path.exists(dest_path):
                        shutil.copy(file_path, dest_path)
                        print(f" -> Copiato con successo: {file_name}")
                        copied_count += 1
                    else:
                        print(f" -> {file_name} era già presente, salto la copia.")
                        copied_count += 1

        print(f"[DATASET] Operazione completata. {copied_count}/{len(target_files)} file pronti in {target_folder}!")
        
        if copied_count < len(target_files):
            print("[AVVISO] Non tutti i file richiesti sono stati trovati. Verifica i nomi sul dataset di Kaggle.")
            
    except Exception as e:
        print(f"[ERRORE] Errore durante il download o il filtraggio da Kaggle: {e}")

def evaluate(model, loader, device):
    """Esegue l'inferenza sui dati non visti (Testing)"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total if total > 0 else 0

def train():
    device = get_device()
    start_time_total = time.time()

    # Configurato per puntare alla cartella montata del container
    data_folder = "/dataset" 
    
    # Esegue il download mirato prima di cercare i file
    download_dataset_via_kaggle(data_folder)

    # 1. SELEZIONE FILE BASATA SUL NOME
    train_files = glob.glob(os.path.join(data_folder, "*-training.parquet"))
    test_files = glob.glob(os.path.join(data_folder, "*-testing.parquet"))
    
    if not train_files:
        print("ERRORE: Nessun file '-training.parquet' trovato!")
        return
    if not test_files:
        print("AVVISO: Nessun file '-testing.parquet' trovato. L'inferenza verrà saltata.")

    print(f"File per Training trovati: {len(train_files)}")
    print(f"File per Testing trovati: {len(test_files)}")

    # 2. CARICAMENTO DATASET (num_workers impostato a 0 per evitare il blocco in emulazione)
    train_dataset = CICDDoSCombinedDataset(train_files, window_size=16)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    test_loader = None
    if test_files:
        test_dataset = CICDDoSCombinedDataset(test_files, window_size=16)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 3. CONFIGURAZIONE MODELLO
    args = Mamba3Config(d_model=32, n_layer=1, d_state=16)
    model = Mamba3MultiClass(args, input_dim=77, num_classes=10).to(device)
    
    # Learning rate ridotto per evitare NaN
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\nInizio Addestramento su: {device}")
    
    epochs = 5
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoca {epoch}")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            if torch.isnan(loss): 
                continue

            loss.backward()
            # Gradient Clipping per stabilità matematica
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())

        # --- FASE DI INFERENZA (TESTING) ---
        if test_loader:
            print(f"\nEsecuzione test su dati non analizzati...")
            acc = evaluate(model, test_loader, device)
            print(f"Risultato Epoca {epoch}: Accuracy Test = {acc:.2f}%")
        
        print(f"Tempo Epoca: {format_time(time.time() - epoch_start)}\n")

    print(f"--- Addestramento Terminato ---")
    print(f"Durata totale: {format_time(time.time() - start_time_total)}")

if __name__ == "__main__":
    train()