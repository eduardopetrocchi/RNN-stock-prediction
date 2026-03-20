import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_model(input_shape: tuple, params: dict):
    """
    Constrói e compila um modelo RNN com camadas LSTM empilhada para regressão.
    A arquitetura é baseada em params.yaml, com camadas LSTM
    configuráveis e Dropout para regularização. A camada de saída
    usa ativação linear para prever valores contínuos de preço.

    Args:
        input_shape (tuple): Forma de entrada (window_size,n_features).
        params (dict): Dicionário de hiperparâmetros


    Returns:
        keras.Model: Modelo Keras compila com MSE e MAE.
    """

    lstm_layers = params["lstm_layers"]
    dropout = params["dropout"]

    layers = []

    for i, layer_cfg in enumerate(lstm_layers):
        if i == 0:
            layers.append(
                LSTM(
                    units=layer_cfg["units"],
                    return_sequences=layer_cfg["return_sequences"],
                    input_shape=input_shape,
                )
            )
        else:
            layers.append(
                LSTM(
                    units=layer_cfg["units"],
                    return_sequences=layer_cfg["return_sequences"],
                )
            )

        layers.append(Dropout(dropout))

    # Camada de saída - regressão (preço contínuo)
    layers.append(Dense(units=1, activation="linear"))

    model = Sequential(layers=layers)

    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(
        optimizer=optimizer, loss="mean_squared_error", metrics=["mean_absolute_error"]
    )

    return model


print("Modelo criado com sucesso!!")
