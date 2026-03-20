import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml

import keras
from keras.models import load_model as keras_load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_params() -> dict:
    """
    Carrega o arquivo completo de parâmetros do projeto (params.yaml).

    Returns:
        dict: Dicionário contendo as configurações globais.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def load_model():
    """
    Carrega o modelo Keras pré-treinado do diretório padrão.

    Returns:
        keras.Model: O modelo RNN carregado.
    """

    model_path = "models/model.keras"
    print(f"Carregando modelo de: {model_path}")

    return keras_load_model(model_path)


def load_test_data() -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Lê os arrays de teste e das datas correspondentes do período de avaliação.

    Returns:
        tuple[np.ndarray,np.ndarray,pd.DatetimeIndex]: (X_test, y_test, dates_test)
    """

    data_dir = "data/processed"

    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    dates_test = pd.to_datetime(
        pd.read_csv(os.path.join(data_dir, "dates_test.csv"))["date"]
    )

    return X_test, y_test, dates_test


def inverse_transform_prices(y_norm: np.ndarray, scaler_target) -> np.ndarray:
    """
    Reverte a normalização MinMaxScaler para retornar os preços em R$."

    Args:
        y_norm (np.ndarray): Valores normalizados (0-1)
        scaler_target (_type_): Scaler ajustado na coluna target durante o pré-processamento.

    Returns:
        np.ndarray: Preços reais na escala original.
    """

    return scaler_target.inverse_transform(y_norm.reshape(-1, 1)).flatten()


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray, scaler_target
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Calcula as métricas de regressão do modelo nos dados de teste.

    Args:
        model (_type_): Modelo RNN treinado.
        X_test (np.ndarray): Sequências de entrada para teste.
        y_test (np.ndarray): Targets normalizados reais.
        scaler_target (_type_): Scaler para inversão da normalização.

    Returns:
        tuple[dict,np.ndarray,np.ndarray]: metrics_dict, y_real_prices, y_pred_prices
    """

    print("Gerando previsões...")
    y_pred_norm = model.predict(X_test).flatten()

    # Inverter normalização
    y_real = inverse_transform_prices(y_test, scaler_target)
    y_pred = inverse_transform_prices(y_pred_norm, scaler_target)

    mae = float(mean_absolute_error(y_real, y_pred))
    mse = float(mean_squared_error(y_real, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_real, y_pred))
    mape = float(np.mean(np.abs((y_real - y_pred) / y_real)) * 100)

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }

    return metrics, y_real, y_pred


def plot_predictions(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex,
    ticker: str,
    plots_dir: Path,
) -> None:
    """
    Gera o gráfico comparando preço real VS. previsão da RNN ao longo do tempo.

    Args:
        y_real (np.ndarray): Preços reais em R$.
        y_pred (np.ndarray): Preços previstos em R$.
        dates (pd.DatetimeIndex): Datas correspondentes ao período de teste.
        ticker (str): Código do ativo (para título do gráfico).
        plots_dir (Path): Diretório de destino para o arquivo PNG.
    """

    plt.figure(figsize=(14, 5))
    plt.plot(dates, y_real, color="red", label="Preço real", linewidth=1.5)
    plt.plot(
        dates, y_pred, color="blue", label="Previsão RNN", linewidth=1.5, linestyle="--"
    )
    plt.title(f"Previsão do preço das ações - {ticker}")
    plt.xlabel("Data")
    plt.ylabel("Preço R$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = plots_dir / "predictions.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de previsões salvo em: {plot_path}")


def plot_training_history(plots_dir: Path) -> None:
    """
    Gera o gráfico de evolução de loss (MSE) e MAE durante o treinamento.

    Lê o histórico salvo em metricos/training.json gerado pelo stage anterior.

    Args:
        plots_dir (Path): Diretório de destino para o arquivo PNG.
    """

    history_path = "metrics/training.json"
    if not os.path.exists(history_path):
        print("Histórico de treinamento não encontrado, pulando plot.")
        return

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train MSE")
    axes[0].plot(history["val_loss"], label="Val MSE")
    axes[0].set_title("Loss (MSE) por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_mae"], label="Train MAE")
    axes[1].plot(history["val_mae"], label="Val MAE")
    axes[1].set_title("MAE por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = plots_dir / "training_history.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Histórico de treinamento salvo em: {plot_path}")


def print_resultados(metrics: dict) -> None:
    """
    Exibe o resumo das métricas de regressão no console.

    Args:
        metrics (dict): Dicionário de métricas calculado em evaluate_model.
    """

    print("\n" + "=" * 60)
    print("RESULTADOS DA AVALIAÇÃO".center(60))
    print("=" * 60)
    print(f"MAE : {metrics['mae']:.4f}")
    print(f"MSE : {metrics['mse']:.4f}")
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"R2 : {metrics['r2']:.4f}")
    print(f"MAPE : {metrics['mape']:.2f}")
    print("=" * 60 + "\n")


def save_metrics(metrics: dict) -> None:
    """
    Exporta o dicionário de métricas para um arquivo JSON.

    Args:
        metrics (dict): Dicionário de métricas de avaliação.
    """

    output_dir = Path("metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "evaluation.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Métricas salvas em: {metrics_path}")


def log_to_mlflow(metrics: dict) -> None:
    """
    Registra métricas e gráficos na última execução ativa do MLflow.

    Args:
        metrics (dict): Dicionário de métricas de avaliação.
    """

    mlflow.set_experiment("rnn_stock_prediction")

    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)

    if not runs.empty:
        run_id = runs.iloc[0].run_id
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(
                {
                    "eval_mae": metrics["mae"],
                    "eval_mse": metrics["mse"],
                    "eval_rmse": metrics["rmse"],
                    "eval_r2": metrics["r2"],
                    "eval_mape": metrics["mape"],
                }
            )
            mlflow.log_artifact("metrics/evaluation.json")
            mlflow.log_artifact("metrics/predictions.png")
            mlflow.log_artifact("metrics/training_history.png")

            print("Métricas e gráficos logados com sucesso no MLFlow.")


def main(overrides: dict = None) -> dict:
    """
    Executa o pipeline completo de avaliação: carga de dados, cálculo de
    métricas, geração de plots e persistência de logs (DVC e MLFow).

    Args:
        overrides (dict, optional): overrides (dict, optional): Parâmetros para sobrescrever os do params.yaml.
            Aceita a chave 'ticker' para nomear corretamente os plots.
            Quando None, usa exclusivamente o params.yaml (fluxo DVC).

    Returns:
        dict: Métricas de avaliação (mae, mse, rmse, r2, mape) e dados para o
              template (dates, y_real, y_pred). Vazio se chamado pelo DVC.
    """

    print("=" * 60)
    print("INICIANDO AVALIAÇÃO DO MODELO".center(60))
    print("=" * 60)

    # 1. Criar diretório de plots
    plots_dir = Path("metrics/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 2. Carregar parâmetros, modelos, dados e scaler
    params = load_params()
    model = load_model()
    if overrides:
        params["data"].update({k: v for k, v in overrides.items() if v is not None})
    X_test, y_test, dates_test = load_test_data()
    scaler_target = joblib.load("models/scaler_target.pkl")
    print(f"Dados carregados: {X_test.shape[0]} amostras de teste\n")

    # 3. Avaliar
    metrics, y_real, y_pred = evaluate_model(model, X_test, y_test, scaler_target)

    # 4. Gerar plots
    plot_predictions(y_real, y_pred, dates_test, params["data"]["ticker"], plots_dir)
    plot_training_history(plots_dir)

    # 5. Resultados e logs
    print_resultados(metrics)
    save_metrics(metrics)
    log_to_mlflow(metrics)

    print("Avaliação concluída com sucesso!!")


if __name__ == "__main__":
    main()
