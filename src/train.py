from keras import Input
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from src.graphics import plot_training_performance

def create_model(input_dim):
    """
    Define y compila la red neuronal con regularización L2, Dropout y métricas recall y precisión.
    """
    model = Sequential([
        # Primera capa densa oculta con Dropout
        Dense(256, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.1),  # Apagamos 10% de las unidades para evitar overfitting
        # Segunda capa densa oculta con Dropout
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.1),
        # Capa de salida con función sigmoide para clasificación binaria
        Dense(1, activation='sigmoid')
    ])
    # Compilar el modelo con métricas recall y precisión
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', Recall(), Precision()]  # Métricas adicionales
    )
    return model

def train_model(features, labels, model_path, scaler_path):
    """
    Entrena el modelo por 15 épocas con EarlyStopping y guarda el modelo entrenado.
    """
    model = create_model(features.shape[1])

    early_stopping = EarlyStopping(
        monitor='val_accuracy',  # Monitorea el recall en validación
        patience=11,            # Detener el entrenamiento si no mejora después de 11 épocas
        restore_best_weights=True,
        mode='max'
    )

    #Ajuste de pesos de clase para balancear las etiquetas
    # - Las clases desbalanceadas se pueden manejar mediante pesos ajustados.
    # - Para las clases minoritarias (lluvia), asignamos un peso mayor para forzar al modelo a prestarle más atención.
    # - En este caso, le damos más peso a la clase positiva (1, lluvia).
    # class_weight = {0: 1, 1: 1.2}  #Peso de 1.2 a la clase 1 (lluvia), 1 a la clase 0 (no lluvia)
    # El modelo le da más importancia a los casos de lluvia, tratando de reducir los falsos negativos.

    class_weight={0: 2, 1: 1} # Ajusta según el balance de clases

    training = model.fit(
        features,
        labels,
        epochs=15,  # Máximo 15 épocas
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        class_weight=class_weight
    )

    # Guardar el modelo entrenado
    model.save(model_path)

    # Graficar el rendimiento del entrenamiento
    plot_training_performance(training, output_path='image/training_performance.png')