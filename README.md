# RNN Stock Prediction — v0.2.0

Previsão de preços de ações da B3 com rede neural recorrente (LSTM), seguindo o padrão MLOps com pipeline versionado via DVC, experimentos rastreados no MLflow/DagsHub e serving via FastAPI.

Na v0.2.0 o pipeline é executado sob demanda para qualquer ticker informado na interface — o modelo é treinado e avaliado na hora, sem necessidade de retreinamento manual.

---

## Changelog

### v0.2.0
- Pipeline executado sob demanda a partir da interface web — qualquer ticker da B3 pode ser informado pelo usuário
- `app.py` orquestra os 3 stages (`load_data → train_model → evaluate_model`) diretamente no POST `/predict`
- `main()` de cada stage aceita `overrides` para sobrescrever `ticker`, `test_start` e `test_end` sem alterar o `params.yaml`
- `evaluate_model.main()` retorna dict com métricas e dados para o template (sem depender de `predict.py`)
- `predict.py` removido (substituído pelo fluxo dinâmico)
- Tooltip do gráfico Plotly com fundo branco e fonte preta

### v0.1.0
- Pipeline fixo para um único ticker definido no `params.yaml`
- Serving via FastAPI com gráfico interativo Plotly e métricas MAE, RMSE, R², MAPE

---

## Estrutura do Projeto

```
rnn-stock-prediction/
├── app/
│   ├── app.py                        # FastAPI — rotas, lifespan e orquestração do pipeline
│   └── templates/
│       ├── index.html                # Landing page
│       ├── predict.html              # Formulário (ticker + período de teste)
│       └── result.html               # Gráfico interativo Plotly + métricas
├── src/
│   ├── data_loading/
│   │   └── load_data.py              # Download yfinance + normalização + sequências
│   ├── model_training/
│   │   ├── model_builder.py          # Arquitetura LSTM
│   │   └── train_model.py            # Treinamento + log MLflow/DagsHub
│   └── model_evaluation/
│       └── evaluate_model.py         # Métricas + plots + retorno para o template
├── data/
│   ├── processed/                    # Artefatos gerados pelo DVC 
│   └── csv/                          # Arquivos CSV brutos 
├── models/                           # Modelo e scalers (não versionar)
├── metrics/                          # JSONs e plots (não versionar)
├── dvc.yaml                          # Pipeline DVC (3 stages)
├── params.yaml                       # Hiperparâmetros e configurações
├── main.py                           # Entrypoint uvicorn
├── pyproject.toml                    # Dependências
├── Dockerfile
├── docker-compose.yml
├── .env                              # Credenciais locais (não versionar)
└── .gitignore
```

---

## Pipeline

O projeto é dividido em 3 stages orquestrados pelo DVC (ou pela app em tempo real):

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

### 1. Clonar o repositório

```bash
git clone https://github.com/eduardopetrocchi/rnn-stock-prediction
cd rnn-stock-prediction
```

### 2. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Instalar dependências

```bash
pip install -e .
```

Ou via Docker Compose (sem precisar de venv local)::

```bash
docker compose up --build    # primeira vez
docker compose up -d         # nas próximas
```

### 4. Configurar credenciais

```bash
cp .env.example .env
# Editar .env com seu usuário e token do DagsHub
```

### 5. Inicializar DVC

```bash
dvc remote add origin https://dagshub.com/eduardopetrocchi/rnn-stock-prediction.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user $MLFLOW_TRACKING_USERNAME
dvc remote modify origin --local password $MLFLOW_TRACKING_PASSWORD
```

---

## Execução

### Via interface web

```bash
python main.py
# http://localhost:8000
```

Informe o ticker (ex: `B3SA3.SA`, `PETR4.SA`, `VALE3.SA`) e o período de teste. O pipeline completo roda automaticamente e exibe o gráfico com preço real vs. previsão LSTM e as métricas de avaliação.

### Via DVC (experimentação de hiperparâmetros)

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
| `data.ticker` | `B3SA3.SA` | Ativo padrão da B3 |
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


## Autor

**Eduardo Petrocchi**  
[linkedin.com/in/eduardopetrocchi](https://www.linkedin.com/in/eduardopetrocchi/)

---

<sub>Template HTML da interface web, README e formatação final assistidos por [Claude](https://claude.ai) (Anthropic).</sub>
