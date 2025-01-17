import matplotlib.pyplot as plt


def plot_training_performance(history, output_path):
    """
    Genera y guarda gráficas de rendimiento del entrenamiento (precisión y pérdida).
    """
    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión al entrenamiento', color='#6ec417')
    plt.plot(history.history['val_accuracy'], label='Precisión a la validación', color='#d17132')
    plt.title('Precisión por Época')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida al entrenamiento', color='#6ec417')
    plt.plot(history.history['val_loss'], label='Pérdida a la validación', color='#d17132')
    plt.title('Pérdida por Época')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_evaluation_metrics(precision, recall, output_path):
    """
    Genera y guarda una gráfica de barras para precisión y recall.
    """
    plt.figure(figsize=(6, 6))
    plt.bar(['Precisión', 'Recall'], [precision, recall], color=['#64ca38', '#e2aa58'])
    plt.ylim(0, 1)
    plt.title('Precisión y Recall del Modelo')
    plt.savefig(output_path)  # Guardar la gráfica
    plt.close()