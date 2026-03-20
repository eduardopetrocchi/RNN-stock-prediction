# Imagem base Python slim
FROM python:3.10-slim

# Diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias para TensorFlow e yfinance
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalar Poetry
#RUN pip install poetry

# Copiar arquivos de dependências primeiro (cache do Docker)
COPY pyproject.toml ./

# Instalar dependências sem criar virtualenv (já estamos no container)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Copiar código fonte
COPY . .

# Criar diretórios necessários para os artefatos do DVC
RUN mkdir -p data/processed models metrics/plots

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expor porta da API
EXPOSE 8000

# Comando padrão — inicia a FastAPI
CMD ["python", "main.py"]