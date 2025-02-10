# Usa un'immagine base con Python 3.5
FROM python:3.5-slim

# Installa gli strumenti di compilazione necessari
RUN apt-get update && apt-get install -y \
    build-essential \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-lang-european \
    dvipng \
    && rm -rf /var/lib/apt/lists/*



# Imposta la cartella di lavoro nel container
WORKDIR /app

# Copia i file del progetto nel container
COPY . /app

# Installa le dipendenze (assumendo che tu abbia un requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Comando di default (se necessario)
CMD ["python", "my_demo.py"]
