import pandas as pd
import numpy as np
import os
import json
import torch
import joblib
import argparse
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from og_procesamiento import create_domain_features
from models import ImprovedNeuralNetwork


def load_model(model_path):
    """Carga el modelo guardado"""
    try:
        print(f'Cargando modelo desde: {model_path}')
        checkpoint = torch.load(model_path)
        
        # Extraer hiperparámetros
        hyperparameters = checkpoint['hyperparameters']
        input_size = hyperparameters['input_size']
        hidden_size = hyperparameters['hidden_size']
        n_layers = hyperparameters['n_layers']
        dropout = hyperparameters['dropout']
        
        # Crear modelo
        model = ImprovedNeuralNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Cargar estado del modelo
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
        
    except Exception as e:
        print(f'Error al cargar el modelo desde {model_path}')
        print(f'Error: {str(e)}')
        raise

def make_predictions(model, X_tensor, batch_size=32):
    """Realiza predicciones en batches para optimizar memoria"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i + batch_size]
            outputs = model(batch_X)
            predictions.append(outputs)
    
    predictions = torch.cat(predictions)
    return (predictions >= 0.5).float().cpu().numpy().flatten()

def main():
    """Función principal"""
    
    # Configurar rutas
    path_raw = os.path.join('.', 'data', 'raw')
    path_processed = os.path.join('.', 'data', 'processed')
    path_final = os.path.join('.', 'data', 'final')
    path_models = os.path.join('.', 'models')
    
    # Asegurar que los directorios existan
    for path in [path_raw, path_processed, path_final, path_models]:
        os.makedirs(path, exist_ok=True)
    
    # Cargar el pipeline de preprocesamiento
    print("\nCargando pipeline de preprocesamiento...")
    try:
        preprocessing_pipeline = joblib.load(os.path.join(path_models, 'preprocessor.joblib'))
        
    except Exception as e:
        print(f"Error al cargar el pipeline: {str(e)}")
        return
    
    # Cargar datos de prueba
    print("Cargando datos de prueba...")
    try:
        df_test = pd.read_csv(os.path.join(path_raw, 'test_infarto.csv'), sep=';')
    except Exception as e:
        print(f"Error al cargar los datos de prueba: {str(e)}")
        return
    
    print(f"Dimensiones originales del dataset: {df_test.shape}")
    
    # Crear características de dominio
    print("\nCreando características de dominio...")
    df_test = create_domain_features(df_test)
    
    # Identificar columnas numéricas excluyendo ID
    numeric_columns = df_test.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_columns = [x for x in numeric_columns if x != 'ID']
    print(f"Número de columnas numéricas: {len(numeric_columns)}")
    
    # Preprocesar datos
    print("\nAplicando preprocesamiento...")
    try:
        X_transformed = preprocessing_pipeline.transform(df_test[numeric_columns])
        print(f"Dimensiones después del preprocesamiento: {X_transformed.shape}")
    except Exception as e:
        print(f"Error en el preprocesamiento: {str(e)}")
        return
    
    # Convertir a tensor
    X_tensor = torch.FloatTensor(X_transformed)
    
    # Cargar el modelo entrenado
    print("\nBuscando modelo entrenado...")
    try:
        model_path = os.path.join(path_models, 'model_final.pt')
        print(f"Cargando modelo: {model_path}")
        model = load_model(model_path)
        if model is None:
            print("No se pudo cargar el modelo. Abortando.")
            return 1
        
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        return
    
    # Realizar predicciones
    print("\nRealizando predicciones...")
    try:
        predictions = make_predictions(model, X_tensor)
        
        # Crear DataFrame de resultados (solo ID y ZSN para el archivo final)
        results_df = pd.DataFrame({
            'ID': df_test['ID'],
            'ZSN': predictions
        })
        
        # Guardar resultados
        output_path = os.path.join(path_final, 'submission_final.csv')
        results_df.to_csv(output_path, index=False)  # Guardamos sin índice y sin separador personalizado
        print(f"\nPredicciones guardadas en: {output_path}")
        
        # Mostrar estadísticas
        print("\nEstadísticas de las predicciones:")
        print(f"Total de casos: {len(predictions)}")
        print(f"Casos positivos: {predictions.sum()} ({(predictions.sum()/len(predictions))*100:.2f}%)")
        print(f"Casos negativos: {len(predictions)-predictions.sum()} ({((len(predictions)-predictions.sum())/len(predictions))*100:.2f}%)")
        
        # Guardar metadatos
        metadata = {
            'total_casos': len(predictions),
            'casos_positivos': int(predictions.sum()),
            'casos_negativos': int(len(predictions)-predictions.sum()),
            'dimensiones_entrada': X_transformed.shape,
            'umbral_prediccion': 0.5
        }
        
        metadata_path = os.path.join(path_final, 'metadata_final.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"\nMetadatos guardados en: {metadata_path}")
        
        return True
    except Exception as e:
        print(f"Error durante la inferencia: {str(e)}")
        return False

if __name__ == "__main__":
    main()
