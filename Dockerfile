# Usa un'immagine Python ufficiale snella
FROM python:3.10-slim

# Imposta variabili d'ambiente per evitare file temporanei e forzare l'output in tempo reale
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installa le dipendenze di sistema necessarie per far girare i grafici (Matplotlib/Seaborn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt-get/lists/*

# Imposta la cartella di lavoro all'interno del container
WORKDIR /app

# Copia prima solo il file delle dipendenze per sfruttare la cache di Docker
COPY requirements.txt .

# Installa le dipendenze Python (se usi solo CPU, puoi installare la versione CPU di torch per risparmiare spazio, altrimenti l'installazione standard va bene)
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto dei file sorgente nella cartella di lavoro
COPY . .

# Crea le cartelle destinate ai dati e ai risultati nel caso in cui non esistessero
RUN mkdir -p /dataset ./output

# Comando di avvio che lancia l'addestramento aggiornato
CMD ["python", "Training_CiC_2.py"]