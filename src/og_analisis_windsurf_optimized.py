import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

# Importar las clases necesarias
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from og_mvp_windsurf_tuned import ImprovedNeuralNetwork, NullColumnDropper

# Definir rutas
path_raw = os.path.join('data', 'raw')
path_processed = os.path.join('data', 'processed')
path_final = os.path.join('data', 'final')
path_models = 'models'

def load_data():
    """Carga los datos procesados"""
    # Cargar datos procesados
    train_file = os.path.join('.', 'data', 'processed', 'train_processed.csv')
    df = pd.read_csv(train_file)
    
    # Separar features y target
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y

def load_model_and_pipeline():
    """Carga el modelo y el pipeline de preprocesamiento"""
    # Cargar modelo
    model_file = os.path.join('.', 'models', 'model_final.pt')
    checkpoint = torch.load(model_file)
    
    # Extraer hiperparámetros
    hyperparameters = checkpoint.get('hyperparameters', {})
    
    # Crear modelo con los hiperparámetros guardados
    model = ImprovedNeuralNetwork(
        input_size=hyperparameters.get('input_size'),
        hidden_size=hyperparameters.get('layer_size'),
        n_layers=hyperparameters.get('n_layers'),
        dropout=hyperparameters.get('dropout')
    )
    
    # Cargar pesos del modelo
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Cargar pipeline
    pipeline_file = os.path.join('.', 'models', f'preprocessor_{timestamp}.joblib')
    preprocessing_pipeline = joblib.load(pipeline_file)
    
    return model, preprocessing_pipeline

def analyze_model_performance(model, preprocessing_pipeline, X, y):
    """Analiza el rendimiento del modelo"""
    try:
        # Preprocesar datos
        X_processed = preprocessing_pipeline.transform(X)
        X_tensor = torch.FloatTensor(X_processed)
        
        # Obtener predicciones
        with torch.no_grad():
            y_pred_proba = model(X_tensor)
            y_pred = (y_pred_proba > 0.5).float().numpy().flatten()
        
        # Calcular métricas
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        
        # Generar reporte de clasificación
        report = classification_report(y, y_pred)
        
        # Crear directorio para resultados si no existe
        os.makedirs('./data/final', exist_ok=True)
        
        # Guardar resultados
        results = {
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'timestamp': timestamp
        }
        
        results_file = os.path.join('.', 'data', 'final', f'model_performance_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Visualizar matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        
        # Guardar gráfico
        plt_file = os.path.join('.', 'data', 'final', 'confusion_matrix.png')
        plt.savefig(plt_file)
        plt.close()
        
        print(f"\nResultados guardados en: {results_file}")
        print(f"Gráfico guardado en: {plt_file}")
        
        return True
    except Exception as e:
        print(f"\nError en el análisis de rendimiento: {str(e)}")
        return False
    
    print("\nResultados del análisis:")
    print(f"F1-Score: {f1:.4f}")
    print("\nMatriz de Confusión:")
    print(cm)
    print("\nReporte de Clasificación:")
    print(report)
    
    return True

def main():
    """Función principal"""
    try:
        print("\nCargando datos...")
        X, y = load_data()
        
        print("\nCargando modelo y pipeline...")
        model, preprocessing_pipeline = load_model_and_pipeline()
        
        print("\nAnalizando rendimiento del modelo...")
        if analyze_model_performance(model, preprocessing_pipeline, X, y):
            print("\nAnálisis completado exitosamente")
            return 0
        else:
            print("\nError en el análisis")
            return 1
            
    except Exception as e:
        print(f"\nError en el proceso: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
