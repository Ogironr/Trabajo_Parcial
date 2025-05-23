import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import joblib

# Definir rutas
path_raw = os.path.join('.', 'data', 'raw')
path_processed = os.path.join('.', 'data', 'processed')
path_final = os.path.join('.', 'data', 'final')
path_models = os.path.join('.', 'models')

# Asegurar que los directorios existan
for path in [path_raw, path_processed, path_final, path_models]:
    os.makedirs(path, exist_ok=True)

# Transformador para eliminar columnas con muchos nulos
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

# Red neuronal mejorada
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
        # Normalización de entrada
        x = self.input_bn(x)
        
        # Primer bloque con conexión residual
        identity1 = self.projection1(x)
        x = self.block1_fc1(x)
        x = self.block1_bn1(x)
        x = self.selu(x)
        x = self.dropout(x)
        x = self.block1_fc2(x)
        x = self.block1_bn2(x)
        x = x + identity1
        x = self.selu(x)
        
        # Segundo bloque con conexión residual
        identity2 = self.projection2(x)
        x = self.block2_fc1(x)
        x = self.block2_bn1(x)
        x = self.selu(x)
        x = self.dropout(x)
        x = self.block2_fc2(x)
        x = self.block2_bn2(x)
        x = x + identity2
        x = self.selu(x)
        
        # Tercer bloque con conexión residual
        identity3 = self.projection3(x)
        x = self.block3_fc1(x)
        x = self.block3_bn1(x)
        x = self.selu(x)
        x = self.dropout(x)
        x = self.block3_fc2(x)
        x = self.block3_bn2(x)
        x = x + identity3
        x = self.selu(x)
        
        # Capa de salida
        x = self.output_fc(x)
        x = self.sigmoid(x)
        
        return x

# Focal Loss para mejor manejo de desbalance
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = -(target * torch.log(pred + 1e-6) + (1 - target) * torch.log(1 - pred + 1e-6))
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_loss = ce_loss * ((1 - pt) ** self.gamma)
        return focal_loss.mean()

# Pérdida combinada
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.focal = FocalLoss(gamma=2.0)
    
    def forward(self, pred, target):
        return self.alpha * self.bce(pred, target) + (1 - self.alpha) * self.focal(pred, target)

def main():
    print("Cargando datos...")
    # Cargar datos
    df_raw = pd.read_csv(os.path.join(path_raw, 'train_infarto.csv'), sep=';')
    
    # Identificar columnas numéricas excluyendo ID y target
    numeric_columns = df_raw.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [x for x in numeric_columns if x not in ['ID', 'ZSN']]
    
    print("Preparando pipeline de preprocesamiento...")
    # Pipeline de preprocesamiento simplificado
    preprocessing_pipeline = Pipeline([
        ('null_dropper', NullColumnDropper(null_threshold=90)),
        ('power_transform', PowerTransformer(method='yeo-johnson')),
        ('robust_scale', RobustScaler())
    ])
    
    # Preparar datos
    X = df_raw[numeric_columns]
    y = df_raw['ZSN']
    
    # División train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Aplicando preprocesamiento...")
    # Aplicar preprocesamiento
    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
    X_test_transformed = preprocessing_pipeline.transform(X_test)
    
    # Guardar pipeline
    joblib.dump(preprocessing_pipeline, os.path.join(path_models, 'preprocessing_pipeline.joblib'))
    
    # Convertir a tensores
    X_train_tensor = torch.FloatTensor(X_train_transformed)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test_transformed)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    print("Inicializando modelo...")
    # Inicializar modelo
    input_size = X_train_transformed.shape[1]
    model = ImprovedNeuralNetwork(input_size)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Parámetros de entrenamiento
    n_epochs = 100
    batch_size = 32
    patience = 10
    best_val_f1 = 0
    patience_counter = 0
    best_epoch = 0
    
    print("Iniciando entrenamiento...")
    # Lista para guardar logs
    training_history = []

    # Entrenamiento
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        # Training loop
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        # Evaluación
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).squeeze()
            y_pred_class = (y_pred > 0.5).float()
            val_f1 = f1_score(y_test_tensor.numpy(), y_pred_class.numpy())
        
        # Guardar logs
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': float(avg_loss),
            'val_f1': float(val_f1)
        }
        training_history.append(epoch_log)
        
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}')
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping en época {epoch}")
            break

    # Guardar historial de entrenamiento
    import json
    with open(os.path.join(path_models, 'training_history.json'), 'w') as f:
        json.dump(training_history, f)
    
    # Guardar historial de entrenamiento
    import json
    history_path = os.path.join(path_models, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f)
    print(f"\nHistorial de entrenamiento guardado en: {history_path}")
    
    print("\nEntrenamiento completado!")
    print(f"Mejor F1-Score: {best_val_f1:.4f} en epoch {best_epoch+1}")
    
    # Evaluación final
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze()
        
        # Encontrar mejor umbral
        thresholds = np.arange(0.1, 1.0, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_class = (y_pred > threshold).float()
            f1 = f1_score(y_test_tensor.numpy(), y_pred_class.numpy())
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Métricas finales
        y_pred_class = (y_pred > best_threshold).float()
        accuracy = accuracy_score(y_test_tensor.numpy(), y_pred_class.numpy())
        precision = precision_score(y_test_tensor.numpy(), y_pred_class.numpy())
        recall = recall_score(y_test_tensor.numpy(), y_pred_class.numpy())
        f1 = f1_score(y_test_tensor.numpy(), y_pred_class.numpy())
        roc_auc = roc_auc_score(y_test_tensor.numpy(), y_pred.numpy())
        
        print("\nResultados finales:")
        print(f"Mejor umbral: {best_threshold:.2f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()
