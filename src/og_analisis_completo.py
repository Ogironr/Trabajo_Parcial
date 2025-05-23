import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import torch

def load_training_history():
    """Carga el historial de entrenamiento desde el modelo final"""
    path_models = os.path.join('.', 'models')
    model_path = os.path.join(path_models, 'model_final.pt')
    
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el archivo {model_path}")
        return None
        
    try:
        checkpoint = torch.load(model_path)
        metrics = checkpoint.get('metrics', {})
        
        if not metrics:
            print("No se encontraron métricas en el modelo")
            return None
            
        # Crear DataFrame con las métricas
        data = {
            'epoch': list(range(1, len(metrics['train_losses']) + 1)),
            'train_loss': metrics['train_losses'],
            'val_f1': metrics['val_f1_scores']
        }
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error al cargar métricas del modelo: {str(e)}")
        return None

def load_inference_results():
    """Carga los resultados de la inferencia y los metadatos"""
    path_final = os.path.join('.', 'data', 'final')
    
    predictions_path = os.path.join(path_final, 'submission_final.csv')
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"No se encontró el archivo de predicciones: {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)
    
    metadata_path = os.path.join(path_final, 'metadata_final.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No se encontró el archivo de metadatos: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return predictions_df, metadata

def load_validation_data():
    """Carga los datos de validación para análisis de rendimiento"""
    try:
        path_processed = os.path.join('.', 'data', 'processed')
        train_file = os.path.join(path_processed, 'train_processed.csv')
        
        if not os.path.exists(train_file):
            print(f"No se encontró el archivo {train_file}")
            return None, None
            
        # Cargar datos
        data = pd.read_csv(train_file)
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Dividir en train/val (igual que en el entrenamiento)
        from sklearn.model_selection import train_test_split
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_val, y_val
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        return None, None

def analyze_training_evolution(logs_df, output_dir):
    """Analiza la evolución del entrenamiento"""
    if len(logs_df) == 0:
        print("Error: No hay datos de entrenamiento disponibles")
        return False
        
    # Verificar columnas necesarias
    required_columns = ['epoch', 'train_loss', 'val_f1']
    for col in required_columns:
        if col not in logs_df.columns:
            print(f"Error: No se encontró la columna '{col}' en los logs")
            return False
            
    plt.style.use('seaborn')
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Evolución del Entrenamiento', fontsize=16)
    
    # 1. Pérdida de entrenamiento y F1-Score
    ax = axes[0]
    ax.plot(logs_df['epoch'], logs_df['train_loss'], label='Pérdida de Entrenamiento', marker='o')
    ax2 = ax.twinx()
    ax2.plot(logs_df['epoch'], logs_df['val_f1'], label='F1-Score (Val)', color='red', marker='s')
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida', color='blue')
    ax2.set_ylabel('F1-Score', color='red')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.grid(True)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 2. Evolución suavizada
    ax = axes[1]
    window = 3
    val_f1_smooth = logs_df['val_f1'].rolling(window=window, min_periods=1).mean()
    train_loss_smooth = logs_df['train_loss'].rolling(window=window, min_periods=1).mean()
    
    ax.plot(logs_df['epoch'], logs_df['val_f1'], 'r.', alpha=0.3, label='F1-Score (Raw)')
    ax.plot(logs_df['epoch'], val_f1_smooth, 'r-', label='F1-Score (Smoothed)')
    ax2 = ax.twinx()
    ax2.plot(logs_df['epoch'], logs_df['train_loss'], 'b.', alpha=0.3, label='Pérdida (Raw)')
    ax2.plot(logs_df['epoch'], train_loss_smooth, 'b-', label='Pérdida (Smoothed)')
    
    ax.set_xlabel('Época')
    ax.set_ylabel('F1-Score', color='red')
    ax2.set_ylabel('Pérdida', color='blue')
    ax.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax.grid(True)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_evolution.png')
    plt.savefig(plot_path)
    plt.close()
    
    return True

def analyze_model_performance(X, y, output_dir):
    """Analiza el rendimiento del modelo en el conjunto de validación"""
    try:
        # Cargar modelo
        model_path = os.path.join('.', 'models', 'model_final.pt')
        if not os.path.exists(model_path):
            print(f"No se encontró el modelo en: {model_path}")
            return False
            
        # Cargar el modelo
        checkpoint = torch.load(model_path)
        model_params = checkpoint['hyperparameters']
        
        # Crear el modelo con los mismos parámetros
        from models import ImprovedNeuralNetwork
        model = ImprovedNeuralNetwork(
            input_size=model_params['input_size'],
            hidden_size=model_params['hidden_size'],
            n_layers=model_params['n_layers'],
            dropout=model_params['dropout']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Convertir datos a tensores
        X_tensor = torch.FloatTensor(X.values)
        
        # Hacer predicciones
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = (outputs >= 0.5).float().numpy()
            
        # Calcular métricas
        f1 = f1_score(y, predictions)
        cm = confusion_matrix(y, predictions)
        report = classification_report(y, predictions)
        
        # Guardar resultados
        results = {
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        results_file = os.path.join(output_dir, 'model_performance.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        # Crear matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        
        plot_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(plot_path)
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error en análisis de rendimiento: {str(e)}")
        return False

def analyze_predictions(predictions_df, metadata, output_dir):
    """Analiza las predicciones del modelo"""
    # Crear distribución de predicciones
    plt.figure(figsize=(10, 6))
    sns.histplot(data=predictions_df['ZSN'], bins=30)
    plt.title('Distribución de Predicciones')
    plt.xlabel('Predicción')
    plt.ylabel('Frecuencia')
    
    plot_path = os.path.join(output_dir, 'predicciones_distribucion.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Calcular estadísticas
    stats = {
        'total_predicciones': len(predictions_df),
        'media_predicciones': predictions_df['ZSN'].mean(),
        'desv_std_predicciones': predictions_df['ZSN'].std(),
        'mediana_predicciones': predictions_df['ZSN'].median(),
        'timestamp_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Generar reporte
    report = f"""# Análisis de Predicciones

## Estadísticas Generales
- Total de predicciones: {stats['total_predicciones']}
- Casos positivos: {metadata['casos_positivos']} ({metadata['casos_positivos']/stats['total_predicciones']*100:.2f}%)
- Casos negativos: {metadata['casos_negativos']} ({metadata['casos_negativos']/stats['total_predicciones']*100:.2f}%)

## Métricas de Predicción
- Media: {stats['media_predicciones']:.4f}
- Desviación estándar: {stats['desv_std_predicciones']:.4f}
- Mediana: {stats['mediana_predicciones']:.4f}

## Detalles del Modelo
- Dimensiones de entrada: {metadata['dimensiones_entrada']}
- Umbral de predicción: {metadata['umbral_prediccion']}

## Visualizaciones
- Distribución de predicciones: predicciones_distribucion.png

## Información del Análisis
- Fecha y hora: {stats['timestamp_analisis']}
"""
    
    report_path = os.path.join(output_dir, 'analisis_predicciones.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return True

def main():
    """Función principal para el análisis completo"""
    try:
        # Configurar directorio de salida
        output_dir = os.path.join('.', 'reports')
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Análisis del entrenamiento
        print("\nAnalizando evolución del entrenamiento...")
        logs_df = load_training_history()
        if logs_df is not None:
            if analyze_training_evolution(logs_df, output_dir):
                print("Análisis de entrenamiento completado")
            else:
                print("Error en análisis de entrenamiento")
        
        # 2. Análisis de rendimiento en validación
        print("\nAnalizando rendimiento del modelo...")
        X, y = load_validation_data()
        if X is not None and y is not None:
            if analyze_model_performance(X, y, output_dir):
                print("Análisis de rendimiento completado")
            else:
                print("Error en análisis de rendimiento")
        
        # 3. Análisis de predicciones
        print("\nAnalizando predicciones...")
        try:
            predictions_df, metadata = load_inference_results()
            if analyze_predictions(predictions_df, metadata, output_dir):
                print("Análisis de predicciones completado")
            else:
                print("Error en análisis de predicciones")
        except Exception as e:
            print(f"Error al cargar resultados de inferencia: {str(e)}")
        
        print("\nAnálisis completo finalizado")
        print(f"Resultados guardados en: {output_dir}")
        return True
        
    except Exception as e:
        print(f"\nError en el análisis: {str(e)}")
        return False

if __name__ == "__main__":
    main()
