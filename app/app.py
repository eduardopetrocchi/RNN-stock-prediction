import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import yaml
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.data_loading.load_data import main as run_load_data
from src.model_training.train_model import main as run_train_model
from src.model_evaluation.evaluate_model import main as run_evaluate_model


# Lifespan - carrega os parâmetros na inicialização.
async def lifespan(app: FastAPI):
    global model, scaler, scaler_target, params

    with open(file="params.yaml") as f:
        params = yaml.safe_load(stream=f)

    print("\nParâmetros carregados com sucesso!!")
    yield


app = FastAPI(title="RNN Stock Prediction", lifespan=lifespan)

TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(object=TEMPLATE_DIR))


# Rotas
@app.get(path="/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(name="index.html", context={"request": request})


@app.get(path="/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    default_ticker = params["data"]["ticker"] if params else "B3SA3.SA"
    return templates.TemplateResponse(
        name="predict.html",
        context={"request": request, "default_ticker": default_ticker},
    )


@app.post(path="/predict", response_class=HTMLResponse)
async def predict_post(
    request: Request,
    ticker: str = Form(default=...),
    start_date: str = Form(default=...),
    end_date: str = Form(default=...),
):
    ticker = ticker.strip().upper()
    overrides = {"ticker": ticker, "test_start": start_date, "test_end": end_date}

    try:
        # 1. Download + normalização + sequências para o ticker informado
        print(f"\n[1/3] Carregando dados: {ticker}: ({start_date} --> {end_date})")
        run_load_data(overrides=overrides)

        # 2. Treina a RNN com os dados do ticker informado
        print(f"\n[2/3] Treinando o modelo para {ticker}")
        run_train_model(overrides=overrides)

        # 3. Avalia e retorna métricas + dados para o template
        print(f"\n[3/3] Avaliando o modelo")
        result = run_evaluate_model(overrides=overrides)

    except Exception as e:
        return templates.TemplateResponse(
            "result.html", {"request": request, "error": str(e)}
        )

    return templates.TemplateResponse(
        name="result.html",
        context={
            "request": request,
            "ticker": result["ticker"],
            "dates": result["dates"],
            "y_real": result["y_real"],
            "y_pred": result["y_pred"],
            "metrics": result["metrics"],
        },
    )
