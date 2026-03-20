import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model as keras_load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import keras


def load_artifacts() -> tuple:
    """
    Carrega o modelo Keras e os dois scalers persistidos na etapa de treinamento.

    Returns:
        tuple: model, scaler_features, scaler_target
    """

    model = keras_load_model(filepath="models/model.keras")
    scaler = joblib.load(filename="models/scaler.pkl")
    scaler_target = joblib.load(filename="models/scaler_target.pkl")

    return model, scaler, scaler_target


def predict_from_yfinance(
    ticker: str,
    start: str,
    end: str,
    model: keras.Model,
    scaler,
    scaler_target,
    window_size: int = 90,
    features: list[str] = None,
) -> dict:
    """
    Baixa dados do yfinance para o período informado e gera previsões com o modelo.

    Utiliza os últimos dias anteriores (window_size) a 'start' para montar
    o lookback completo da primeira previsão.

    Args:
        ticker (str): Código do ativo (ex: 'PETR4.SA')
        start (str): Data de ínicio do período de teste (YYYY-MM-DD).
        end (str): Data de fim do périodo de teste (YYYY-MM-DD).
        model (keras.Model): Modelo Keras carregado.
        scaler (_type_): MinMaxScaler ajustado nas features (todas as colunas).
        scaler_target (_type_): MinMaxScaler ajustado apenas no 'Close'.
        window_size (int, optional): Tamanho da janela de lookback.
        features (list[str], optional): Colunas usadas como features.

    Returns:
        dict: Dicionário com preços reais, previstos, datas e métricas.
        Retorna {'error': str} em caso de falha.
    """

    if features is None:
        features = ["Open", "High", "Low", "Close", "Volume"]

    target_idx = features.index("Close")

    # Período expandido para garantir lookback completo
    df_full = yf.download(
        tickers=ticker,
        start=pd.Timestamp(start) - pd.tseries.offsets.BDay(window_size + 10),
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(df_full.columns, pd.MultiIndex):
        df_full.columns = df_full.columns.get_level_values(0)

    df_full.dropna(inplace=True)

    if len(df_full) < window_size + 1:
        return {
            "error": f"Dados insuficientes para o ticker {ticker} no período informado."
        }

    # Separar janela de lookback e período de teste
    df_test = df_full[df_full.index >= start][features]

    if len(df_test) == 0:
        return {"error": f"Nenhum dado encontrado para '{ticker}' a partir de {start}."}

    # Normalizar a série completa
    full_values = df_full[features].values
    full_norm = scaler.transform(full_values)

    # Montar sequências de teste com lookback
    test_start_idx = len(df_full) - len(df_test)
    X_test, y_test_norm = [], []
    for i in range(test_start_idx, len(full_norm)):
        X_test.append(full_norm[i - window_size : i])
        y_test_norm.append(full_norm[i, target_idx])

    X_test = np.array(object=X_test)

    # Gerar previsões
    y_pred_norm = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler_target.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_real = scaler_target.inverse_transform(
        np.array(object=y_test_norm).reshape(-1, 1)
    ).flatten()

    # Calcular métricas
    mae = float(mean_absolute_error(y_true=y_real, y_pred=y_pred))
    mse = float(mean_squared_error(y_true=y_real, y_pred=y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true=y_real, y_pred=y_pred))
    mape = float(np.mean(np.abs((y_real - y_pred) / y_real)) * 100)

    return {
        "ticker": ticker,
        "dates": [str(object=d.date()) for d in df_test.index],
        "y_real": [round(number=float(v), ndigits=4) for v in y_real],
        "y_pred": [round(number=float(v), ndigits=4) for v in y_pred],
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mape": round(mape, 2),
        },
    }
