# RNN Stock Prediction

Previsão de preços de ações da B3 com rede neural recorrente (LSTM), seguindo o padrão MLOps do projeto com pipeline versionado via DVC, experimentos rastreados no MLflow/DagsHub e serving via FastAPI.

---

## Estrutura do Projeto

```
rnn-stock-prediction/
├── app/
│   ├── app.py                        # FastAPI — rotas e lifespan
│   └── templates/
│       ├── index.html                # Landing page
│       ├── predict.html              # Formulário (ticker + período)
│       └── result.html               # Gráfico interativo + métricas
├── src/
│   ├── data_loading/
│   │   └── load_data.py              # Download yfinance + normalização + sequências
│   ├── model_training/
│   │   ├── model_builder.py          # Arquitetura LSTM
│   │   └── train_model.py            # Treinamento + log MLflow/DagsHub
│   ├── model_evaluation/
│   │   └── evaluate_model.py         # Métricas + plots
│   └── predicting/               
│       └── predict.py                # Inferência reutilizável pela app
├── data/
│   ├── processed/                    # Artefatos gerados pelo DVC 
│   └── csv/                          # Arquivos CSV brutos 
├── models/                           # Modelo e scalers 
├── metrics/                          # JSONs e plots 
├── dvc.yaml                          # Pipeline DVC (3 stages)
├── params.yaml                       # Hiperparâmetros e configurações
├── main.py                           # Entrypoint uvicorn
├── pyproject.toml                    # Dependências
├── Dockerfile
├── .env                              # Credenciais locais 
├── .env.example                      # Exemplo para as credenciais locais 
└── .gitignore
```

---

## Pipeline

O projeto é dividido em 3 stages orquestrados pelo DVC:

```
load_data  ──►  train_model  ──►  evaluate_model
```

| Stage | Arquivo | Responsabilidade |
|---|---|---|
| `load_data` | `src/data_loading/load_data.py` | Download via yfinance, normalização MinMax, construção das sequências de janela deslizante |
| `train_model` | `src/model_training/train_model.py` | Treinamento da RNN LSTM com EarlyStopping, ReduceLROnPlateau e ModelCheckpoint. Log no MLflow/DagsHub |
| `evaluate_model` | `src/model_evaluation/evaluate_model.py` | Cálculo de MAE, RMSE, R², MAPE. Geração de gráficos e log das métricas no MLflow |

---

## Configuração

### 1. Clonar e instalar dependências

```bash
git clone https://dagshub.com/eduardopetrocchi/rnn-stock-prediction
cd rnn-stock-prediction

pip install poetry
poetry install
```

Ou via Docker Compose:
 
```bash
docker compose up --build    # primeira vez
docker compose up -d         # nas próximas
```

### 2. Configurar credenciais

```bash
cp .env.example .env
# Editar .env com seu usuário e token do DagsHub
```

### 3. Inicializar DVC

```bash
dvc remote add origin https://dagshub.com/eduardopetrocchi/rnn-stock-prediction.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user $MLFLOW_TRACKING_USERNAME
dvc remote modify origin --local password $MLFLOW_TRACKING_PASSWORD
```

---

## Execução do Pipeline

```bash
# Executa todos os stages (ou só os afetados por mudanças)
dvc repro

# Verifica as métricas
dvc metrics show

# Compara com a run anterior
dvc metrics diff
```

---

## Experimentos

Para testar variações de hiperparâmetros sem editar o `params.yaml` manualmente:

```bash
# Sobrescreve um parâmetro pontualmente
dvc exp run --set-param model.dropout=0.2
dvc exp run --set-param model.learning_rate=0.0005
dvc exp run --set-param data.window_size=60

# Visualiza todos os experimentos lado a lado
dvc exp show

# Aplica o melhor experimento ao branch principal
dvc exp apply <exp-name>
```

Todos os experimentos são sincronizados automaticamente com o MLflow no DagsHub.

---

## Serving (FastAPI)

```bash
python main.py
# http://localhost:8000
```

A interface permite informar qualquer ticker da B3 (ex: `PETR4.SA`, `VALE3.SA`, `ITUB4.SA`) e um período de datas. O modelo retorna o gráfico interativo com preço real vs. previsão LSTM e as métricas de avaliação.

---

## Arquitetura do Modelo

```
Input (window_size=90, features=5)
    │
    ▼
LSTM(100) → Dropout(0.3)
    │
LSTM(50)  → Dropout(0.3)
    │
LSTM(50)  → Dropout(0.3)
    │
LSTM(50)  → Dropout(0.3)
    │
Dense(1, activation='linear')
    │
    ▼
Preço previsto (R$)
```

**Features de entrada:** Open, High, Low, Close, Volume  
**Target:** Close (preço de fechamento)  
**Normalização:** MinMaxScaler(0, 1) — scaler de features e scaler de target independentes  

---

## Parâmetros Principais

| Parâmetro | Valor padrão | Descrição |
|---|---|---|
| `data.ticker` | `B3SA3.SA` | Ativo da B3 |
| `data.window_size` | `90` | Dias de lookback |
| `model.dropout` | `0.3` | Taxa de dropout |
| `model.learning_rate` | `0.001` | Learning rate Adam |
| `model.epochs` | `100` | Máximo de épocas (com EarlyStopping) |
| `model.batch_size` | `32` | Batch size |
| `model.patience` | `10` | Paciência do EarlyStopping |

---

## Stack

| Camada | Tecnologia |
|---|---|
| Modelo | TensorFlow / Keras (LSTM) |
| Dados | yfinance |
| Pipeline | DVC |
| Experimentos | MLflow + DagsHub |
| Serving | FastAPI + Jinja2 |
| Visualização | Plotly.js |
| Containerização | Docker |