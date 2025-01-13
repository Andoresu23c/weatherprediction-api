from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from datetime import datetime

app = Flask(__name__)

# Ruta al modelo, escalador y codificador guardados
model_path = "models/clima_model.h5"
scaler_path = "models/scaler.pkl"
encoder_path = "models/encoder.pkl"

# Cargar el modelo entrenado
model = load_model(model_path)

# Cargar el escalador y el codificador
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint para predecir lluvia basado en datos enviados como JSON.
    """
    global date
    input_data = request.get_json()

    # Validar que se envíen los datos necesarios
    if "data" not in input_data or "country" not in input_data or "date" not in input_data:
        return jsonify({"error": "Faltan datos en la solicitud"}), 400

    try:
        # Preparar características numéricas
        data = np.array([input_data["data"]])  # Convertir los datos en un array de NumPy
        data_normalized = scaler.transform(data)  # Normalizar los datos

        # Validar la fecha
        try:
            date = input_data["date"]
            parsed_date = datetime.strptime(date, "%Y-%m-%d")  # Verificar formato y validez
        except ValueError:
            return jsonify({"error": f"La fecha '{date}' no es válida. Use el formato 'YYYY-MM-DD'."}), 400

        # Validar si el país está entre las categorías conocidas
        country = input_data["country"]
        if country not in encoder.categories_[0]:
            return jsonify({"error": f"El país '{country}' no está en los datos de entrenamiento. "
                                     f"Por favor, use uno de: {encoder.categories_[0].tolist()}"}), 400

        # Codificar el país
        country_encoded = encoder.transform([[country]])  # Codificar país con OneHotEncoder
        # Codificar la fecha
        date = datetime.strptime(input_data["date"], "%Y-%m-%d")
        date_features = np.array([[date.month, date.day]])  # Extraer mes y día

        # Concatenar características
        features_final = np.hstack([data_normalized, country_encoded, date_features])

        # Realizar la predicción
        prediction = model.predict(features_final)
        prediction_value = prediction[0][0]  # Obtener el valor de la predicción

        # Clasificar según la predicción
        predicted_class = "Lluvia" if prediction_value > 0.5 else "No Lluvia"

        # Responder con los resultados
        return jsonify({
            "country": country,
            "date": input_data["date"],
            "prediction": float(prediction_value),
            "type": predicted_class
        })

    except Exception as e:
        return jsonify({"error": f"Error procesando la solicitud: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
