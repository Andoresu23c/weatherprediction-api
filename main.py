from config import DATASET_PATH, MODEL_PATH, SCALER_PATH
from src.preprocess_data import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    """
    Función principal que coordina el preprocesamiento de datos,
    entrenamiento del modelo y evaluación de su desempeño.
    """
    print("=== INICIO DEL PROCESO ===")
    data = pd.read_csv(DATASET_PATH)

    #1. Preprocesar los datos
    print("Preprocesando los datos...")
    features, labels = preprocess_data(data)

    #2. Dividir los datos en conjuntos de entrenamiento y prueba
    print("Dividiendo los datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    #3. Entrenar el modelo
    print("Entrenando el modelo...")
    train_model(X_train, y_train, MODEL_PATH, SCALER_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    #4. Evaluar el modelo
    precision, recall = evaluate_model(MODEL_PATH, X_test, y_test, threshold=0.5)
    print(f"Precisión del modelo: {precision:.2f}")
    print(f"Recall del modelo: {recall:.2f}") #Taza de verdaderos positivos

if __name__ == "__main__":
    main()