from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.models import load_model

def evaluate_model(model_path, X_test, y_test, threshold=0.5):
    try:
        print("Cargando el modelo...")
        model = load_model(model_path)

        print("Realizando predicciones...")
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > threshold).astype("int32")

        print("Calculando métricas...")
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        return precision, recall

    except Exception as e:
        print(f"Error durante la evaluación: {e}")
        return None, None
