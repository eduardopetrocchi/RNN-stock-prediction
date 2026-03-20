import os
import joblib
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def load_params() -> dict:
    """
    Carrega as configurações de dados a partir do arquivo YAML de parâmetros.

    Returns:
        dict: Dicionário contendo as chaves de configuração sob a seção 'data'.
    """
    with open(file="params.yaml", mode="r") as f:
        params = yaml.safe_load(stream=f)

    return params["data"]


def download_data(params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza o download das séries históricas via yfinance para treino e teste.

    Args:down
        params (dict): Parâmetros de dados do params.yaml

    Returns:
        tuple[pd.DataFrame,pd.DataFrame]: DataFrames de treino e teste com as colunas definidas em params['features'].
    """

    ticker = params["ticker"]
    features = params["features"]

    print(
        f"Baixando dados de treinamento: {ticker} ({params['train_start']} -> {params['train_end']})"
    )
    df_train = yf.download(
        tickers=ticker,
        start=params["train_start"],
        end=params["train_end"],
        auto_adjust=True,
    )

    print(
        f"Baixando dados de teste: {ticker} ({params['test_start']} -> {params['test_end']})"
    )
    df_test = yf.download(
        tickers=ticker,
        start=params["test_start"],
        end=params["test_end"],
        auto_adjust=True,
    )

    for df in [df_train, df_test]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    csv_dir = "data/csv"
    os.makedirs(name=csv_dir, exist_ok=True)

    df_train.to_csv(path_or_buf=os.path.join(csv_dir, "df_train.csv"), index=True)
    df_test.to_csv(path_or_buf=os.path.join(csv_dir, "df_test.csv"), index=True)
    print(f"Arquivos salvos:")
    print(f"data/csv/df_train.csv")
    print(f"data/csv/df_test.csv")

    return df_train[features], df_test[features]


def normalize_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame, params: dict
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Aplica normalização MinMaxScaler às features e ao target de forma independente.
    O scaler de features é ajustado apenas no treino e aplicado em ambos os conjuntos.
    O scalres de target é ajustado apenas na coluna alvo do treino.

    Args:
        df_train (pd.DataFrame): Dados brutos de treino.
        df_test (pd.DataFrame): Dados brutos de teste.
        params (dict): Parâmetros contendo 'target' e 'features'.

    Returns:
        tuple[np.ndarray,np.ndarray,MinMaxScaler,MinMaxScaler]: (train_norm,test_norm,scaler_features,scaler_target)
    """

    target_col = params["target"]
    target_idx = params["features"].index(target_col)

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    train_norm = scaler_features.fit_transform(df_train)
    test_norm = scaler_features.transform(df_test)

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit_transform(df_train[[target_col]].values)

    return train_norm, test_norm, scaler_features, scaler_target


def build_sequences(
    train_norm: np.ndarray, test_norm: np.ndarray, df_test: pd.DataFrame, params: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Constroí as sequências de entrada (janela deslizante) para treino e teste.
    Para o teste, concate os últimos 'window_size' dias do treino para garantir
    que as primeiras previsões tenham lookback completo.

    Args:
        train_norm (np.ndarray): Dados de treino normalizados.
        test_norm (np.ndarray): Dados de teste normalizados
        df_test (pd.DataFrame): DataFrame original de teste (para extrair datas).
        params (dict): Parâmetros contendo 'window_size', 'features' e 'target'.

    Returns:
        tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,pd.DatetimeIndex]: (X_train, X_test, y_train, y_test, dates_test)
    """

    window = params["window_size"]
    n_feat = len(params["features"])
    target_idx = params["features"].index(params["target"])

    # Sequências de treino
    X_train, y_train = [], []
    for i in range(window, len(train_norm)):
        X_train.append(train_norm[i - window : i, :n_feat])
        y_train.append(train_norm[i, target_idx])

    # Sequências de teste (lookback inclui final do treino)
    full_series = np.concatenate([train_norm[-window:], test_norm], axis=0)
    X_test, y_test = [], []
    for i in range(window, len(full_series)):
        X_test.append(full_series[i - window : i, :n_feat])
        y_test.append(test_norm[i - window, target_idx])

    X_train = np.array(object=X_train)
    y_train = np.array(object=y_train)
    X_test = np.array(object=X_test)
    y_test = np.array(object=y_test)

    dates_test = df_test.index

    return X_train, X_test, y_train, y_test, dates_test


def save_artifacts(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    dates_test: pd.DatetimeIndex,
    scaler_features: MinMaxScaler,
    scaler_target: MinMaxScaler,
) -> None:
    """
    Persiste os arrays numpy, as datas de teste e os scalers em disco.

    Args:
        X_train (np.ndarray): Sequências de entrada para treino.
        X_test (np.ndarray): Sequências de entrada para teste.
        y_train (np.ndarray): Targets de treino normalizados.
        y_test (np.ndarray): Targets de teste normalizados.
        dates_test (pd.DatetimeIndex): Índice de datas do período de teste.
        scaler_features (MinMaxScaler): Scaler ajustado nas features completas.
        scaler_target (MinMaxScaler): Scaler ajustado apenas no target (Close).
    """

    data_dir = "data/processed"
    model_dir = "models"

    os.makedirs(name=data_dir, exist_ok=True)
    os.makedirs(name=model_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "X_train.npy"), arr=X_train)
    np.save(os.path.join(data_dir, "X_test.npy"), arr=X_test)
    np.save(os.path.join(data_dir, "y_train.npy"), arr=y_train)
    np.save(os.path.join(data_dir, "y_test.npy"), arr=y_test)

    pd.DataFrame(data={"date": dates_test}).to_csv(
        path_or_buf=os.path.join(data_dir, "dates_test.csv"), index=False
    )

    joblib.dump(value=scaler_features, filename=os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(
        value=scaler_target, filename=os.path.join(model_dir, "scaler_target.pkl")
    )


def main(overrides: dict = None) -> None:
    """
    Executa o pipeline completo de download, normalização e geração de sequências.
    Orquestra desde o download via yfinance até a exportação dos artefatos
    prontos para o treinamento da RNN.

    Args:
        overrides (dict, optional): Parâmetros para sobrescrever params.yaml.
        Aceita as chaves 'ticker','test_start' e 'test_end'.
        Quando None, usa exclusivamente o params.yaml (fluxo DVC).
    """

    # 1. Carregar parâmetros
    params = load_params()
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})

    # 2. Download dos dados
    df_train, df_test = download_data(params=params)

    # 3. Normalizar
    train_norm, test_norm, scaler_features, scaler_target = normalize_data(
        df_train=df_train, df_test=df_test, params=params
    )

    # 4. Construir sequências
    X_train, X_test, y_train, y_test, dates_test = build_sequences(
        train_norm=train_norm, test_norm=test_norm, df_test=df_test, params=params
    )

    # 5. Salvar artefatos
    save_artifacts(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        dates_test=dates_test,
        scaler_features=scaler_features,
        scaler_target=scaler_target,
    )

    print("Dados salvos com sucesso!!")
    print(f"X_train: {X_train.shape} y_train{y_train.shape}")
    print(f"X_test{X_test.shape} y_test{y_test.shape}")


if __name__ == "__main__":
    main()
