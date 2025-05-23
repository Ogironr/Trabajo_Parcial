import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin

# Definir los transformadores personalizados (deben ser iguales a los del entrenamiento)
class NullColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, null_threshold=90):
        self.null_threshold = null_threshold
        self.columns_to_drop_ = None
    
    def fit(self, X, y=None):
        null_percentage = (X.isnull().sum() / len(X)) * 100
        self.columns_to_drop_ = null_percentage[null_percentage > self.null_threshold].index.tolist()
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        if self.columns_to_drop_:
            X_transformed = X_transformed.drop(columns=self.columns_to_drop_)
        X_transformed = X_transformed.fillna(0)
        return X_transformed

class InteractionFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_target_features=18204):
        self.n_target_features = n_target_features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        n_original = X.shape[1]
        features_needed = self.n_target_features - n_original
        
        if features_needed <= 0:
            return X
        
        X_new = X.copy()
        interaction_count = 0
        
        # Agregar interacciones hasta alcanzar el número objetivo
        for i in range(n_original):
            if interaction_count >= features_needed:
                break
            for j in range(i+1, n_original):
                if interaction_count >= features_needed:
                    break
                X_new = np.column_stack([X_new, X[:, i] * X[:, j]])
                interaction_count += 1
        
        return X_new

# Definir rutas
path_raw = os.path.join('.', 'data', 'raw')
path_processed = os.path.join('.', 'data', 'processed')
path_final = os.path.join('.', 'data', 'final')
path_models = os.path.join('.', 'models')

# Cargar el pipeline de preprocesamiento
print("Cargando pipeline de preprocesamiento...")
preprocessing_pipeline = joblib.load(os.path.join(path_models, 'preprocessing_pipeline.joblib'))

# Cargar datos de prueba
print("Cargando datos de prueba...")
df_test = pd.read_csv(os.path.join(path_raw, 'test_infarto.csv'), sep=';')
print(f"Dimensiones originales del dataset: {df_test.shape}")

# Identificar columnas numéricas excluyendo ID
numeric_columns = df_test.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns = [x for x in numeric_columns if x != 'ID']
print(f"Número de columnas numéricas originales: {len(numeric_columns)}")

# Crear DataFrame con las columnas numéricas (no aplicamos fillna porque el pipeline lo hará)
X = df_test[numeric_columns].copy()

# Aplicar el pipeline de preprocesamiento
print("\nAplicando pipeline de preprocesamiento...")
X_transformed = preprocessing_pipeline.transform(df_test[numeric_columns])
print(f"Dimensiones después del pipeline: {X_transformed.shape}")

# Convertir a DataFrame
feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]
X = pd.DataFrame(X_transformed, columns=feature_names)

print(f"Dimensiones finales: {X.shape}")

# Verificar que tengamos el número correcto de features
expected_features = 18204  # Número de features que espera el modelo
if X.shape[1] != expected_features:
    raise ValueError(f"Error: Número incorrecto de features. Esperado: {expected_features}, Obtenido: {X.shape[1]}")

print(f"Features actuales: {current_features}")
print(f"Features necesarias: {target_features}")
print(f"Features adicionales requeridas: {features_needed}")
# Verificar cuántas features adicionales necesitamos
features_needed = target_features - X.shape[1]

if features_needed > 0:
    print(f"\nNecesitamos agregar {features_needed} features adicionales")
    # Agregar interacciones selectivas hasta alcanzar el número objetivo
    interaction_count = 0
    for i in range(len(numeric_columns)):
        if interaction_count >= features_needed:
            break
        for j in range(i+1, len(numeric_columns)):
            if interaction_count >= features_needed:
                break
            col1, col2 = numeric_columns[i], numeric_columns[j]
            interaction_name = f'interaction_{i}_{j}'
            X[interaction_name] = X_scaled[:, i] * X_scaled[:, j]
            interaction_count += 1
    print(f"Se agregaron {interaction_count} interacciones")

# Verificación final de dimensiones
print(f"\nDimensiones finales: {X.shape}")
if X.shape[1] != target_features:
    raise ValueError(f"Error: Número incorrecto de features. Esperado: {target_features}, Obtenido: {X.shape[1]}")

# Convertir a tensor
X_oot_tensor = torch.FloatTensor(X.values)
print(f"Dimensiones del tensor final: {X_oot_tensor.shape}")

# Guardar el tensor para referencia
torch.save(X_oot_tensor, os.path.join(path_processed, 'X_oot_tensor.pt'))

print(f"Dimensiones del tensor de entrada: {X_oot_tensor.shape}")

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Definir la red neuronal mejorada
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNeuralNetwork, self).__init__()
        
        # Capas principales
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # Primer bloque
        self.block1_fc1 = nn.Linear(input_size, 256)
        self.block1_bn1 = nn.BatchNorm1d(256)
        self.block1_fc2 = nn.Linear(256, 256)
        self.block1_bn2 = nn.BatchNorm1d(256)
        self.projection1 = nn.Linear(input_size, 256)
        
        # Segundo bloque
        self.block2_fc1 = nn.Linear(256, 128)
        self.block2_bn1 = nn.BatchNorm1d(128)
        self.block2_fc2 = nn.Linear(128, 128)
        self.block2_bn2 = nn.BatchNorm1d(128)
        self.projection2 = nn.Linear(256, 128)
        
        # Tercer bloque
        self.block3_fc1 = nn.Linear(128, 64)
        self.block3_bn1 = nn.BatchNorm1d(64)
        self.block3_fc2 = nn.Linear(64, 64)
        self.block3_bn2 = nn.BatchNorm1d(64)
        self.projection3 = nn.Linear(128, 64)
        
        # Capa de salida
        self.output_fc = nn.Linear(64, 1)
        
        # Dropout adaptativo
        self.dropout = nn.Dropout(0.3)
        
        # Activaciones
        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Normalización inicial
        x = self.input_bn(x)
        
        # Primer bloque con skip connection
        identity1 = self.projection1(x)
        x = self.block1_fc1(x)
        x = self.block1_bn1(x)
        x = self.selu(x)
        x = self.dropout(x)
        x = self.block1_fc2(x)
        x = self.block1_bn2(x)
        x = self.selu(x)
        x = x + identity1
        
        # Segundo bloque con skip connection
        identity2 = self.projection2(x)
        x = self.block2_fc1(x)
        x = self.block2_bn1(x)
        x = self.selu(x)
        x = self.dropout(x)
        x = self.block2_fc2(x)
        x = self.block2_bn2(x)
        x = self.selu(x)
        x = x + identity2
        
        # Tercer bloque con skip connection
        identity3 = self.projection3(x)
        x = self.block3_fc1(x)
        x = self.block3_bn1(x)
        x = self.selu(x)
        x = self.dropout(x)
        x = self.block3_fc2(x)
        x = self.block3_bn2(x)
        x = self.selu(x)
        x = x + identity3
        
        # Capa de salida
        x = self.output_fc(x)
        x = self.sigmoid(x)
        return x


# Inicializar la red neuronal con la misma arquitectura
input_size = X_oot.shape[1]
modelo_inferencia = ImprovedNeuralNetwork(input_size)

# Cargar el modelo entrenado
try:
    checkpoint_path = os.path.join(path_models, 'best_model_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        modelo_inferencia.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modelo cargado exitosamente de {checkpoint_path}")
        print(f"F1-score del modelo cargado: {checkpoint['val_f1']:.4f}")
    else:
        raise FileNotFoundError(f"No se encontró el archivo de checkpoint en {checkpoint_path}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    raise

# Poner el modelo en modo evaluación
modelo_inferencia.eval()



# Cargar parámetros del modelo (si existen)
try:
    parameters_path = os.path.join(path_models, 'parameters.json')
    if os.path.exists(parameters_path):
        with open(parameters_path, 'r') as f:
            parameters = json.load(f)
        best_threshold = parameters.get("best_threshold", 0.5)
        print(f"Usando threshold: {best_threshold}")
    else:
        best_threshold = 0.5
        print(f"No se encontró archivo de parámetros. Usando threshold por defecto: {best_threshold}")
except Exception as e:
    print(f"Error al cargar parámetros: {e}")
    best_threshold = 0.5

# Realizar predicciones
with torch.no_grad():
    predictions = modelo_inferencia(X_oot_tensor)
    predictions = predictions.squeeze().numpy()
    predictions_binary = (predictions > best_threshold).astype(int)

# Preparar DataFrame de resultados
df_submit = df_test[['ID']].copy()
df_submit['ZSN'] = predictions_binary

# Guardar resultados
fecha = '20250508'  # Puedes ajustar la fecha según necesites
output_path = os.path.join(path_final, f'submission_{fecha}.csv')
df_submit.to_csv(output_path, index=False)
print(f"\nPredicciones guardadas en: {output_path}")

# Mostrar estadísticas de las predicciones
print("\nEstadísticas de las predicciones:")
print(f"Total de casos analizados: {len(predictions)}")
print(f"Casos predichos como positivos: {predictions_binary.sum()}")
print(f"Porcentaje de casos positivos: {(predictions_binary.sum() / len(predictions) * 100):.2f}%")

# Convertir datos a tensor de PyTorch si aún no lo es
if not isinstance(X_oot_tensor, torch.Tensor):
    X_oot_tensor = torch.FloatTensor(X_oot_scaled)

# Cargar el modelo entrenado
try:
    # Intentar cargar el checkpoint más reciente primero
    checkpoint_path = os.path.join(path_models, 'best_model_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        # Si no existe, intentar con el nombre alternativo
        checkpoint_path = os.path.join(path_models, 'modelo_entrenado_20250507.pth')
    
    if os.path.exists(checkpoint_path):
        print(f"Cargando modelo desde: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Obtener las dimensiones del modelo guardado
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            input_size = checkpoint['model_state_dict']['input_bn.weight'].shape[0]
            state_dict = checkpoint['model_state_dict']
        else:
            input_size = checkpoint['input_bn.weight'].shape[0]
            state_dict = checkpoint
            
        print(f"Dimensiones del modelo guardado: {input_size}")
        
        # Verificar si las dimensiones coinciden
        if input_size != X_oot.shape[1]:
            raise ValueError(f"Error de dimensiones: El modelo espera {input_size} características pero los datos tienen {X_oot.shape[1]}")
            
        # Reinicializar el modelo con las dimensiones correctas
        modelo_inferencia = ImprovedNeuralNetwork(input_size)
        modelo_inferencia.load_state_dict(state_dict)
        
        if isinstance(checkpoint, dict) and 'val_f1' in checkpoint:
            print(f"F1-score del modelo cargado: {checkpoint['val_f1']:.4f}")
            
        print(f"Modelo cargado exitosamente")
    else:
        raise FileNotFoundError(f"No se encontró ningún archivo de modelo en {path_models}")

    # Poner el modelo en modo evaluación
    modelo_inferencia.eval()
    
    # Realizar predicciones
    with torch.no_grad():
        predictions = modelo_inferencia(X_oot_tensor)
        predictions = predictions.squeeze().numpy()
        predictions_binary = (predictions > best_threshold).astype(int)
    
    # Preparar DataFrame de resultados
    df_submit = df_test[['ID']].copy()
    df_submit['ZSN'] = predictions_binary
    
    # Guardar resultados
    fecha = '20250508'
    output_path = os.path.join(path_final, f'submission_{fecha}.csv')
    df_submit.to_csv(output_path, index=False)
    print(f"\nPredicciones guardadas en: {output_path}")
    
    # Mostrar estadísticas
    print("\nEstadísticas de las predicciones:")
    print(f"Total de casos analizados: {len(predictions)}")
    print(f"Casos predichos como positivos: {predictions_binary.sum()}")
    print(f"Porcentaje de casos positivos: {(predictions_binary.sum() / len(predictions) * 100):.2f}%")

except Exception as e:
    print(f"Error durante la inferencia: {e}")
    raise

df_submit['ZSN'].value_counts()