import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def preprocess_data(data):
    """
    Preprocesa los datos, codifica países y fechas, y genera las etiquetas.
    """
    # Características numéricas
    features = data[['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                     'precipitation_sum', 'wind_speed_10m_max']].values

    # Etiquetas: 1 si hay precipitación, 0 si no hay
    labels = (data['precipitation_sum'] > 0).astype(int)

    # Normalizar características numéricas
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)

    # Codificar países (OneHotEncoder)
    encoder = OneHotEncoder(sparse_output=False)
    countries_encoded = encoder.fit_transform(data[['country']])

    # Extraer mes y día de la columna 'date'
    data['month'] = pd.to_datetime(data['date']).dt.month
    data['day'] = pd.to_datetime(data['date']).dt.day

    # Crear las características de fecha
    date_features = data[['month', 'day']].values

    # Concatenar todas las características (numéricas + países + fecha)
    features_final = np.hstack([features_normalized, countries_encoded, date_features])

    # Guardar el escalador y el codificador
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("models/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    return features_final, labels
