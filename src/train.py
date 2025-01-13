import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model(input_dim):
    """
    Define y compila la red neuronal.
    """
    #Función de activación ReLU
    model = Sequential([
        Dense(32, input_dim=input_dim, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#Función para entrenar el modelo 15 epocas,
#además, de cuardar el modelo entrenado en la ruta
def train_model(features, labels, model_path, scaler_path):
    model = create_model(features.shape[1])
    model.fit(features, labels, epochs=15, batch_size=32, validation_split=0.2)
    model.save(model_path)

