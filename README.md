# Weather Prediction Project

Este proyecto utiliza un modelo de red neuronal para predecir si lloverá, basado en datos meteorológicos como temperaturas, precipitación y velocidad del viento. A continuación, se describe cómo inicializar el proyecto, entrenar el modelo, y ejecutar el API.

---

## **Requisitos Previos**

### **Instalaciones necesarias:**
1. Python 3.8 o superior.
2. Entorno virtual (opcional, pero recomendado).
3. Dependencias del proyecto.

### **Clonar el repositorio:**
Clona este proyecto en tu máquina local.

```bash
git clone <URL del repositorio>
cd <nombre_del_proyecto>
```

---

## **Configuración del Entorno**

### **1. Crear un entorno virtual**

Si deseas trabajar en un entorno virtual:

```bash
python -m venv venv
source venv/bin/activate # En Linux/Mac
venv\Scripts\activate   # En Windows
```

### **2. Instalar dependencias**

Ejecuta el siguiente comando para instalar todas las dependencias necesarias que se encuentran en `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## **Entrenamiento del Modelo**

### **1. Preprocesamiento de Datos**

El script `preprocess_data.py` carga y preprocesa los datos desde el archivo CSV proporcionado en la ruta configurada en `config.py`. Esto incluye:
- Filtrar por ciudad (por ejemplo, Quito).
- Normalizar los datos.
- Generar las etiquetas.

Asegúrate de colocar tu archivo CSV en la carpeta `data` con el nombre `LA_daily_climate.csv` o actualizar la ruta en `config.py`.

### **2. Entrenamiento**

Ejecuta la función `main` para entrenar el modelo:

```bash
python main.py
```

Esto generará:
- Un archivo del modelo guardado en `models/clima_model.h5`.
- Un archivo del escalador normalizado en `models/scaler.pkl`.

---

## **Inicialización del API**

El proyecto incluye un API Flask para realizar predicciones basadas en el modelo entrenado.

### **1. Ejecutar el servidor Flask**

Ejecuta el script `predict_api.py` para inicializar el servidor:

```bash
python predict_api.py
```

El servidor estará disponible en `http://127.0.0.1:5000`.

---

## **Uso del API**

### **1. Endpoint: POST /predict**

Este endpoint acepta datos meteorológicos en formato JSON y devuelve una predicción.

- **URL:** `http://127.0.0.1:5000/predict`
- **Método:** POST
- **Headers:**
  - `Content-Type: application/json`
- **Body:**

Formato del JSON:
```json
{
    "data": [temperature_max, temperature_min, temperature_mean, precipitation, wind_speed]
}
```

### **Ejemplo de Solicitud:**

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"data": [30, 20, 25, 0.5, 5.0]}'
```

### **Ejemplo de Respuesta:**

```json
{
    "prediction": 0.9841372966766357,
    "type": "Lluvia"
}
```

---

## **Notas Adicionales**

1. **Archivos Clave:**
   - `main.py`: Entrena el modelo.
   - `predict_api.py`: Inicializa el servidor API.
   - `config.py`: Contiene las rutas de los archivos y configuraciones clave.

2. **Ruta del CSV:**
   Asegúrate de que el archivo de datos esté en la ruta configurada en `config.py`.

3. **Puertos del Servidor:**
   Cambia el puerto en `app.run()` si necesitas usar uno diferente.

---
