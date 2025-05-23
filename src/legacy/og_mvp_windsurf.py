import pandas as pd
import numpy as np
import os


path_raw = os.path.join('.', 'data', 'raw')
path_processed = os.path.join('.', 'data', 'processed')
path_final = os.path.join('.', 'data', 'final')
path_models = os.path.join('.', 'models')


# Crear transformador personalizado para eliminar columnas con muchos nulos
from sklearn.base import BaseEstimator, TransformerMixin

class NullColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, null_threshold=90):
        self.null_threshold = null_threshold
        self.columns_to_drop_ = None
    
    def fit(self, X, y=None):
        # Calcular porcentaje de nulos por columna
        null_percentage = (X.isnull().sum() / len(X)) * 100
        # Identificar columnas a eliminar
        self.columns_to_drop_ = null_percentage[null_percentage > self.null_threshold].index.tolist()
        return self
    
    def transform(self, X):
        # Crear una copia para no modificar el original
        X_transformed = X.copy()
        # Eliminar las columnas identificadas
        if self.columns_to_drop_:
            X_transformed = X_transformed.drop(columns=self.columns_to_drop_)
        # Imputar valores nulos restantes con 0
        X_transformed = X_transformed.fillna(0)
        return X_transformed

# Cargar datos
df_raw = pd.read_csv(os.path.join(path_raw, 'train_infarto.csv'), sep = ';')


# from sklearn.preprocessing import MinMaxScaler
# # Crear una instancia de MinMaxScaler
# scaler = MinMaxScaler()
# numeric_columns = df_raw.select_dtypes(include=['float64', 'int64']).columns
# df_raw[numeric_columns] = scaler.fit_transform(df_raw[numeric_columns])


# Preprocesamiento mejorado usando Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Identificar columnas numéricas excluyendo ID y target
numeric_columns = df_raw.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = [x for x in numeric_columns if x not in ['ID', 'ZSN']]

# Crear transformador para las interacciones
class InteractionFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_target_features=12099):
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

# Definir el número objetivo de características
TARGET_FEATURES = 18204

# Crear pipeline de preprocesamiento completo
preprocessing_pipeline = Pipeline([
    ('null_dropper', NullColumnDropper(null_threshold=90)),
    ('power_transform', PowerTransformer(method='yeo-johnson')),
    ('robust_scale', RobustScaler()),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('interactions', InteractionFeatures(n_target_features=TARGET_FEATURES))
])

# Aplicar pipeline a las columnas numéricas
X_transformed = preprocessing_pipeline.fit_transform(df_raw[numeric_columns])

# Crear nombres para las features polinomiales
poly_feature_names = [f'poly_{i}' for i in range(X_transformed.shape[1])]
df_transformed = pd.DataFrame(X_transformed, columns=poly_feature_names)

# Crear interacciones entre variables transformadas
for i in range(len(numeric_columns)):
    for j in range(i+1, len(numeric_columns)):
        interaction_name = f'interaction_{i}_{j}'
        df_transformed[interaction_name] = X_transformed[:, i] * X_transformed[:, j]

# Guardar el pipeline para usar en inferencia
import joblib
os.makedirs(path_models, exist_ok=True)
joblib.dump(preprocessing_pipeline, os.path.join(path_models, 'preprocessing_pipeline.joblib'))

# Actualizar df_raw con las nuevas features
df_raw = pd.concat([df_raw[['ID', 'ZSN']], df_transformed], axis=1)



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Separar características (X) y etiquetas (y)
target = 'ZSN'
X = df_raw.drop(columns=[target, 'ID'])  # Reemplaza 'target' con el nombre de tu columna objetivo
y = df_raw[target]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



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
input_size = X_train.shape[1]
model = ImprovedNeuralNetwork(input_size)

# Mover el modelo a GPU si está disponible
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')
# model = model.to(device)

# Parámetros de entrenamiento
epochs = 250  # Aumentado para permitir mejor convergencia
batch_size = 64  # Batch size más grande para mejor estimación del gradiente

# Función de pérdida combinada
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.focal = FocalLoss(gamma=2.0)

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        focal_loss = self.focal(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * focal_loss

# Focal Loss para mejor manejo de desbalance
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        # Asegurarse de que las dimensiones coincidan
        if len(pred.shape) == 0:
            pred = pred.unsqueeze(0)
        if len(target.shape) == 0:
            target = target.unsqueeze(0)
            
        # Calcular focal loss
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# Usar la pérdida combinada
criterion = CombinedLoss()

# Optimizador mejorado
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0005,  # Learning rate inicial más bajo
    weight_decay=0.1,  # Mayor regularización L2
    betas=(0.9, 0.999),
    eps=1e-8
)

# Learning rate scheduler con warmup
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.005,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.2,  # 20% del entrenamiento para warmup
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=1000.0
)
best_val_loss = float('inf')
best_model_state = None
patience = 10
no_improve_count = 0

# Asegurarse de que el directorio de modelos existe y esté vacío
if not os.path.exists(path_models):
    os.makedirs(path_models)

# Crear conjunto de validación
val_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Inicializar variables de seguimiento
best_val_loss = float('inf')
best_val_f1 = 0.0
best_epoch = 0
best_model_state = None
checkpoint = None

for epoch in range(epochs):
    # Entrenamiento
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        # Asegurarnos que las dimensiones coincidan
        if len(outputs.shape) == 0:
            outputs = outputs.unsqueeze(0)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validación
    model.eval()
    val_loss = 0
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X).squeeze()
            if len(outputs.shape) == 0:
                outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            val_predictions.extend(outputs.numpy())
            val_targets.extend(batch_y.numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Calcular métricas de validación
    val_predictions = np.array(val_predictions)
    val_targets = np.array(val_targets)
    val_f1 = f1_score(val_targets, (val_predictions > 0.5).astype(int))
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")
    
    # Guardar el mejor modelo basado en F1-score
    if val_f1 > best_val_f1:
        print(f"Nuevo mejor F1 encontrado: {val_f1:.4f} > {best_val_f1:.4f}. Guardando modelo...")
        best_val_f1 = val_f1
        best_model_state = model.state_dict().copy()
        best_epoch = epoch
        no_improve_count = 0
        
        # Guardar el modelo en disco
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': best_val_f1
        }
        
        checkpoint_path = os.path.join(path_models, 'best_model_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Modelo guardado en: {checkpoint_path}")
    else:
        no_improve_count += 1
    
    if no_improve_count >= patience:
        print(f"Early stopping triggered! Best F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
        break



# Cargar el mejor modelo
checkpoint_path = os.path.join(path_models, 'best_model_checkpoint.pth')
if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation F1: {checkpoint['val_f1']:.4f}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print(f"\nAdvertencia: No se encontró checkpoint en {checkpoint_path}")
    print("Usando el modelo actual sin cargar checkpoint.\n")
    checkpoint = {
        'epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': best_val_f1
    }


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





