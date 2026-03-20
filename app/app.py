import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import yaml
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.predicting.predict import load_artifacts, predict_from_yfinance


# Lifespan - carrega o modelo e scalers uma única vez na inicizalização
async def lifespan(app: FastAPI):
    global model, scaler, scaler_target, params

    with open(file="params.yaml") as f:
        params = yaml.safe_load(stream=f)

    try:
        model, scaler, scaler_target = load_artifacts()
        print("\nModelo e scalers carregados com sucesso!!")
    except Exception as e:
        print(f"\nErro ao carregar artefatos: {e}")
        model = scaler = scaler_target = None

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
        name="predict.html", context={"request": request, "default_ticker": default_ticker}
    )


@app.post(path="/predict", response_class=HTMLResponse)
async def predict_post(
    request: Request,
    ticker: str = Form(default=...),
    start_date: str = Form(default=...),
    end_date: str = Form(default=...),
):
    if model is None:
        return templates.TemplateResponse(
            name="result.html",
            context={
                "request": request,
                "error": "Modelo não carregado. Execute o pipeline de treinamento primeiro",
            },
        )

    result = predict_from_yfinance(
        ticker=ticker.strip().upper(),
        start=start_date,
        end=end_date,
        model=model,
        scaler=scaler,
        scaler_target=scaler_target,
        window_size=params["data"]["window_size"],
        features=params["data"]["features"],
    )

    if "error" in result:
        return templates.TemplateResponse(
            name="result.html", context={"request": request, "error": result["error"]}
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
