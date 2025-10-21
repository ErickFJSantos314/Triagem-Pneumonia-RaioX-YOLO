# Usa uma imagem base estável do Python 3.10 (compatível com PyTorch e Ultralytics)
FROM python:3.10-slim

# Instala as bibliotecas de sistema necessárias para o OpenCV/libGL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    # Instala o git-lfs caso você use push via terminal futuramente
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia e instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código (index.py e os modelos .pt)
COPY . /app

# Define a porta e o comando de inicialização
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "index.py", "--server.port=8501", "--server.address=0.0.0.0"]
