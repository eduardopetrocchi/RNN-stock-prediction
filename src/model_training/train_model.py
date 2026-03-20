import json
import os
from pathlib import Path

import dagshub
import mlflow
import mlflow.keras
import numpy as np
import yaml
from dotenv import load_dotenv
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.model_training.model_build import build_model

load_dotenv()


def load_params() -> dict:
    """
    Carrega as configurações de dados a partir do arquivo YAML de parâmetros.

    Returns:
        dict: Dicionário contendo as chaves de configuração sob a seção 'data'.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    return params


def load_processed_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega os arrays numpy gerados na etapa de processamento.

    Returns:
        tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]: (X_train, X_test, y_train, y_test) prontos para o treinamento.
    """

    data_dir = "data/processed"

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    return X_train, X_test, y_train, y_test


def setup_mlflow_experiment() -> dict:
    """
    Configura a integração entre o DagsHub, MlFlow e DVC.
    Inicializa o rastreamento remoto e define se a execução atual deve ser
    tratada como uma "Nested Run" (execução aninhada) dentro de um experimento DVC.

    Returns:
        dict: Argumentos extras (parent_run_id, run_name, nested) para mlflow.start_run.
    """

    dagshub.init(
        repo_owner=os.getenv("MLFLOW_TRACKING_USERNAME"),
        repo_name="rnn-stock-prediction",
    )
    mlflow.set_experiment("rnn_stock_prediction")
    mlflow.keras.autolog()

    is_dvc_experiment = os.getenv("DVC_EXP_NAME") is not None
    extra_args = {}

    if is_dvc_experiment:
        runs = mlflow.search_runs(
            experiment_ids=[os.getenv("MLFLOW_EXPERIMENT_ID")],
            filter_string="tags.dvc_exp='True'",
            order_by=["start_time DESC"],
        )

        if runs.empty:
            with mlflow.start_run() as parent_run:
                mlflow.set_tag("dvc_exp", True)
                parent_run_id = parent_run.info.run_id
        else:
            parent_run_id = runs.iloc[0].run_id

        extra_args = {
            "parent_run_id": parent_run_id,
            "run_name": os.getenv("DVC_EXP_NAME"),
            "nested": True,
        }

    return extra_args


def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    params: dict,
) -> tuple:
    """
    Orquestra o ciclo de vida do treinamento da RNN.

    Gerencia a criação do modelo LSTM, o ajuste com callbacks (EarlyStopping,
    ReduceLROnPlateau, ModelCheckpoint), a avaliação final e o log de parâmetros
    e artefatos no servidos MLFlow.

    Args:
        X_train (np.ndarray): Sequências de entrada para treino (samples, window, features).
        X_test (np.ndarray): Sequências de entrada para validação (samples, window, features).
        y_train (np.ndarray): Targes normalizados de treino.
        y_test (np.ndarray): Targets normalizados de validação/teste.
        params (dict): Dicionário completo de parâmetros (data e model).

    Returns:
        tuple: (model, history, test_loss,test_mae)
    """

    model_params = params["model"]
    extra_args = setup_mlflow_experiment()

    with mlflow.start_run(**extra_args):

        # Logar os parâmetros
        mlflow.log_params(
            {
                "ticker": params["data"]["ticker"],
                "train_start": params["data"]["train_start"],
                "train_end": params["data"]["train_end"],
                "test_start": params["data"]["test_start"],
                "test_end": params["data"]["test_end"],
                "window_size": params["data"]["window_size"],
                "n_features": len(params["data"]["features"]),
                "target": params["data"]["target"],
                "epochs": model_params["epochs"],
                "batch_size": model_params["batch_size"],
                "dropout": model_params["dropout"],
                "learning_rate": model_params["learning_rate"],
                "n_lstm_layers": len(model_params["lstm_layers"]),
            }
        )

        # Logar artefatos de pré-processamento
        for scaler_file in ["models/scaler.pkl", "models/scaler_target.pkl"]:
            if os.path.exists(scaler_file):
                mlflow.log_artifact(scaler_file)

        # Criar o modelo
        print("\n" + "=" * 60)
        print("Criando o modelo RNN")
        print("=" * 60)

        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_model(input_shape=input_shape, params=model_params)

        print(f"Input shape: {input_shape}")
        print("Arquitetura do modelo:")
        model.summary()

        # Callbacks
        es = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-10,
            patience=model_params["patience"],
            restore_best_weights=True,
            verbose=1,
        )
        rlr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=model_params["lr_factor"],
            patience=model_params["patience"],
            verbose=1,
        )
        mcp = ModelCheckpoint(
            filepath="models/model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )

        # Treinar
        print("\n" + "=" * 60)
        print("Treinando o modelo")
        print("=" * 60)

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=model_params["epochs"],
            batch_size=model_params["batch_size"],
            callbacks=[es, rlr, mcp],
        )

        # Avaliar
        print("\n" + "=" * 60)
        print("Avaliando o modelo")
        print("=" * 60)

        test_loss, test_mae = model.evaluate(X_test, y_test)

        print("\n" + "=" * 60)
        print("Resultados finais:")
        print(f"Test loss: (MSE): {test_loss:.5f}")
        print(f"Test MAE: {test_mae:.5f}")
        print("=" * 60)

        # Logar métricas finais
        mlflow.log_metric("test_loss_mse", test_loss)
        mlflow.log_metric("test_mae", test_mae)

        # Salvar e logar artefatos
        save_training_artifacts(model, history, test_loss, test_mae)

        mlflow.log_artifact("models/model.keras")
        mlflow.log_artifact("metrics/training.json")

        return model, history, test_loss, test_mae


def save_training_artifacts(
    model,
    history,
    test_loss: float,
    test_mae: float,
) -> None:
    """
    Persiste o modelo treinado e o histórico de métricas em disco.

    Args:
        model: Modelo treinado a ser salvo
        history: Histórico de treinamento.
        test_loss (float): MSE final no conjunto de teste.
        test_mae (float): MAE final no conjunto de teste.
    """

    models_dir = Path("models")
    metrics_dir = Path("metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.keras"
    model.save(model_path)
    print(f"\nModelo salvo em: {model_path}")

    training_metrics = {
        "train_loss": [float(x) for x in history.history["loss"]],
        "train_mae": [float(x) for x in history.history["mean_absolute_error"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "val_mae": [float(x) for x in history.history["val_mean_absolute_error"]],
        "final_test_mse": float(test_loss),
        "final_test_mae": float(test_mae),
        "epochs_trained": len(history.history["loss"]),
    }

    metrics_path = metrics_dir / "training.json"
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f)
    print(f"Métricas salvas em: {metrics_path}")


def main() -> None:
    """
    Ponto de entrada do pipeline de treinamento.

    Orquestra o carregamento de parâmetros e dados, executa o treinamento
    e exibe o sumário final de performance no console.
    """
    print("\n" + "=" * 60)
    print("INICIANDO TREINAMENTO DA RNN...")
    print("=" * 60)

    # 1. Carregar parâmetros
    print("\n1. Carregando parâmetros...")
    params = load_params()
    print("Parâmetros carregados!!")

    # 2. Carregador dados processos
    print("\n2. Carregando dados processados...")
    X_train, X_test, y_train, y_test = load_processed_data()
    print("Dados carregados com sucesso!!")
    print(f" -Treino: X{X_train.shape} y{y_train.shape}")
    print(f" -Teste: X{X_test.shape} y{y_test.shape}")

    # 3. Treinar o modelo
    print("\n3. Treinando o modelo RNN...")
    model, history, test_loss, test_mae = train_model(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, params=params
    )

    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!!")
    print("=" * 60)
    print(f"MSE final no teste: {test_loss:.6f}")
    print(f"MAE final no teste: {test_mae:.6f}")


if __name__ == "__main__":
    main()
