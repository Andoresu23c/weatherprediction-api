from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.models import load_model

def evaluate_model(model_path, X_test, y_test):
    """
    Carga el modelo, realiza predicciones y evalúa precisión y recall.
    """
    model = load_model(model_path)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return precision, recall
