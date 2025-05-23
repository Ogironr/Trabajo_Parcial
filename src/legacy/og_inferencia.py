import pandas as pd
import numpy as np
import os

path_raw = os.path.join('.', 'data', 'raw')
path_processed = os.path.join('.', 'data', 'processed')
path_final = os.path.join('.', 'data', 'final')
path_models = os.path.join('.', 'models')

df_test = pd.read_csv(os.path.join(path_raw, 'test_infarto.csv'), sep = ';')
df_test.fillna(0, inplace = True)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numeric_columns = df_test.select_dtypes(include=['float64', 'int64']).columns
df_test[numeric_columns] = scaler.fit_transform(df_test[numeric_columns])

X_oot = df_test.drop(columns=['ID'])  # Reemplaza 'target' con el nombre de tu columna objetivo


# Cargar desde JSON
with open('variables_finales.json', 'r') as archivo_json:
    columnas_cargadas = json.load(archivo_json)

X_oot = X_oot[columnas_cargadas]

# # Escalar los datos
# scaler = StandardScaler()
# X_oot = scaler.fit_transform(X_oot)

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


# Inicializar la red neuronal
input_size = X_oot.shape[1]
modelo_cargado = ImprovedNeuralNetwork(input_size)

# Convertir los datos a tensores de PyTorch
X_oot_tensor = torch.tensor(X_oot.values, dtype=torch.float32)

# Cargar los pesos guardados en el modelo
modelo_cargado.load_state_dict(torch.load(os.path.join(path_models, 'modelo_entrenado_20250507.pth')))

# Establecer el modelo en modo de evaluación
modelo_cargado.eval()



import json
import os
parameters_path = os.path.join(path_models, 'parameters.json')
with open(parameters_path, 'r') as f:
    parameters = json.load(f)

best_threshold = parameters["best_threshold"]



y_pred_oot = modelo_cargado(X_oot_tensor).squeeze()
y_pred_class_oot = (y_pred_oot > best_threshold).float()
df_test['ZSN'] = y_pred_class_oot



df_submit = df_test[['ID','ZSN']]
df_submit['ZSN'] = df_submit['ZSN'].astype(int)
fecha = '20250507'
df_submit.to_csv(os.path.join(path_final, 'submission_{}.csv'.format(fecha)), index = False)

df_submit['ZSN'].value_counts()