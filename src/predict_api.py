from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Ruta al modelo y al escalador guardados
model_path = "models/clima_model.h5"
scaler_path = "models/scaler.pkl"

# Cargar el modelo entrenado
model = load_model(model_path)

# Cargar el escalador guardado
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint para predecir lluvia basado en datos enviados como JSON.
    """
    input_data = request.get_json()

    # Validar que se envíen los datos
    if "data" not in input_data:
        return jsonify({"error": "Faltan datos en la solicitud"}), 400

    # Preparar los datos para predicción
    data = np.array([input_data["data"]])  # Convertir los datos en un array de NumPy
    data_normalized = scaler.transform(data)  # Normalizar los datos

    # Realizar la predicción
    prediction = model.predict(data_normalized)  # Obtener la probabilidad
    predicted_class = (prediction > 0.5).astype("int32")  # Clasificar (1 para lluvia, 0 para no lluvia)

    # Responder con los resultados
    return jsonify({
        "prediction": float(prediction[0][0]),
        "type": "Lluvia" if predicted_class[0][0] == 1 else "No Lluvia"
    })

if __name__ == "__main__":
    app.run(debug=True)
