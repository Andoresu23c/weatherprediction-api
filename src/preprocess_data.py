import pickle
from typing import BinaryIO

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import DATASET_PATH


def load_data():
    """
    Carga el archivo CSV desde la ruta especificada en config.py.
    """
    data = pd.read_csv(DATASET_PATH, sep=",")
    return data

def preprocess_data():
    """
    Carga, filtra por la ciudad de Quito y el rango de años 2023-2024,
    limpia y normaliza los datos.
    """
    # Cargar los datos
    data = load_data()

    # Convertir la columna 'date' a formato datetime
    data['date'] = pd.to_datetime(data['date'])

    # Filtrar por país y rango de años
    data = data[(data['country'] == 'Ecuador') & (data['date'].dt.year.isin([2023, 2024]))]

    # Verificar si los datos están vacíos
    if data.empty:
        raise ValueError("No hay datos disponibles para Ecuador en el rango de años 2023-2024.")

    # Eliminar valores nulos
    data = data.dropna()

    # Seleccionar características y etiquetas
    features = data[['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                     'precipitation_sum', 'wind_speed_10m_max']]
    labels = data['precipitation_sum'] > 0  # Etiqueta binaria: 1 si hubo precipitación, 0 si no.

    # Normalizar las características
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)
    # Guardar el escalador para su reutilización en predicciones
    with open("models/scaler.pkl", "wb") as f:
        f: BinaryIO
        pickle.dump(scaler, f)
    return features_normalized, labels