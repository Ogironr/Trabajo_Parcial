import pandas as pd
import numpy as np
import os


path_raw = os.path.join('.', 'data', 'raw')
path_processed = os.path.join('.', 'data', 'processed')
path_final = os.path.join('.', 'data', 'final')
path_models = os.path.join('.', 'models')


#------------------

df_train = pd.read_csv(os.path.join(path_raw, 'train_infarto.csv'), sep = ',')
df_train.columns
df_train.rename(columns = {'target': 'ZSN'}, inplace = True)
df_train.to_csv(os.path.join(path_processed, 'train_infarto.csv'), sep = ';', index = False)

df_test = pd.read_csv(os.path.join(path_raw, 'test_infarto.csv'), sep = ';')
df_test.columns
df_test.rename(columns = {'target': 'ZSN'}, inplace = True)
df_test.to_csv(os.path.join(path_processed, 'test_infarto.csv'), index = False, sep = ';')

#------------------

df_raw = pd.read_csv(os.path.join(path_raw, 'train_infarto.csv'), sep = ';')

df_raw.isnull().sum().sort_values(ascending = False)

null_percentage = (df_raw.isnull().sum() / len(df_raw)) * 100
null_percentage_sorted = null_percentage.sort_values(ascending=False)
columns_to_keep = null_percentage[null_percentage > 90].index

df_raw.drop(columns=columns_to_keep, inplace=True)
# Inputación de valores nulos
df_raw.fillna(0, inplace = True)


# from sklearn.preprocessing import MinMaxScaler
# # Crear una instancia de MinMaxScaler
# scaler = MinMaxScaler()
# numeric_columns = df_raw.select_dtypes(include=['float64', 'int64']).columns
# df_raw[numeric_columns] = scaler.fit_transform(df_raw[numeric_columns])


# Normalizar los datos (añadir antes del entrenamiento)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_columns = df_raw.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = [x for x in numeric_columns if x not in ['ID', 'ZSN']]
df_raw[numeric_columns] = scaler.fit_transform(df_raw[numeric_columns])



import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Separar características (X) y etiquetas (y)
target = 'ZSN'
X = df_raw.drop(columns=[target, 'ID'])  # Reemplaza 'target' con el nombre de tu columna objetivo
y = df_raw[target]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar en JSON
columnas_lista = X_train.columns.tolist()
with open('variables_finales.json', 'w') as archivo_json:
    json.dump(columnas_lista, archivo_json)

print(X_train.shape)
print(X_test.shape)

X_train = X_train.values
X_test = X_test.values

# Convertir los datos a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Crear un DataLoader para el conjunto de entrenamiento
from torch.utils.data import WeightedRandomSampler
# Calcular los pesos para cada clase
class_counts = np.bincount(y_train.to_numpy().astype(int))
total_samples = len(y_train)
class_weights = 1.0 / class_counts
weights = class_weights[y_train.to_numpy().astype(int)]

# Crear el sampler
sampler = WeightedRandomSampler(weights, len(weights))

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=sampler)


# Definir la red neuronal mejorada
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNeuralNetwork, self).__init__()
        # 1er capa
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  # Reducir dropout para permitir mejor aprendizaje
        
        # 2da capa
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        # 3er capa
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        # 4ta capa
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        
        # Capa de salida
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x



# Inicializar la red neuronal
input_size = X_train.shape[1]
model = ImprovedNeuralNetwork(input_size)

# Mover el modelo a GPU si está disponible
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')
# model = model.to(device)

# Calcular los pesos de las clases
class_counts = np.bincount(y_train.to_numpy().astype(int))
total_samples = len(y_train)
# Convertir todo a tensores de PyTorch desde el principio
class_counts_tensor = torch.FloatTensor(class_counts)
weights = total_samples / (len(class_counts) * class_counts_tensor)
# Asegurarnos que los pesos sean del tipo correcto
class_weights = weights.float()

# Definir la función de pérdida y el optimizador
criterion = nn.BCELoss()
# criterion = nn.BCELoss(weight=class_weights)
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Añadir regularización L2
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.02) #0.01

# Implementar reducción de tasa de aprendizaje
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)


# Entrenar la red neuronal
epochs = 100
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


# Evaluar el modelo
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze()
    y_pred_class = (y_pred > 0.5).float()
    accuracy = (y_pred_class == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"Accuracy: {accuracy:.4f}")


from sklearn.metrics import f1_score
# Probar diferentes umbrales
thresholds = np.arange(0.1, 1.0, 0.05)
best_f1 = 0
best_threshold = 0

for threshold in thresholds:
    y_pred_class = (y_pred > threshold).float()
    f1 = f1_score(y_test_tensor.cpu().numpy(), y_pred_class.cpu().numpy())

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Mejor umbral: {best_threshold:.2f}, F1-Score: {best_f1:.4f}")



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# Evaluar el modelo
model.eval()

with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze()
    y_pred_class = (y_pred > best_threshold).float()

    # Calcular la puntuación F1
    f1 = f1_score(y_test_tensor.cpu().numpy(), y_pred_class.cpu().numpy())
    print(f"F1-Score: {f1:.4f}")

    # Calcular el área bajo la curva ROC (AUC)
    auc = roc_auc_score(y_test_tensor.cpu().numpy(), y_pred.cpu().numpy())
    print(f"AUC: {auc:.4f}")

    # Calcular el Gini
    gini = 2 * auc - 1
    print(f"GINI: {gini:.4f}")










import json
import os

# Parámetros a guardar
parameters = {
    "best_threshold": best_threshold,
    "best_f1": f1,
    "auc": auc,
    "gini": gini
}

# Ruta para guardar el archivo JSON
parameters_path = os.path.join(path_models, 'parameters.json')

# Guardar los parámetros en un archivo JSON
with open(parameters_path, 'w') as f:
    json.dump(parameters, f)

# Guardar el modelo entrenado
torch.save(model.state_dict(), os.path.join(path_models, 'modelo_entrenado_20250507.pth'))





