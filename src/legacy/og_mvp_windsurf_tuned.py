import pandas as pd
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import joblib
import optuna
import logging
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.base import BaseEstimator, TransformerMixin
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer

# Definir paths
path_raw = os.path.join('.', 'data', 'raw')
path_processed = os.path.join('.', 'data', 'processed')
path_final = os.path.join('.', 'data', 'final')
path_models = os.path.join('.', 'models')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Definir rutas
path_raw = os.path.join('.', 'data', 'raw')
path_processed = os.path.join('.', 'data', 'processed')
path_final = os.path.join('.', 'data', 'final')
path_models = os.path.join('.', 'models')

# Asegurar que los directorios existan
for path in [path_raw, path_processed, path_final, path_models]:
    os.makedirs(path, exist_ok=True)

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

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=3, dropout=0.5):
        super(ImprovedNeuralNetwork, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # Crear capas ocultas
        self.layers = nn.ModuleList()
        
        current_size = input_size
        for _ in range(n_layers):
            layer_block = nn.Sequential(
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.layers.append(layer_block)
            current_size = hidden_size
        
        # Capa de salida
        self.output_fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.input_bn(x)
        
        # Pasar por cada capa
        for layer in self.layers:
            x = layer(x)
        
        # Capa de salida
        x = self.output_fc(x)
        x = self.sigmoid(x)
        return x

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = -(target * torch.log(pred + 1e-6) + (1 - target) * torch.log(1 - pred + 1e-6))
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_loss = ce_loss * ((1 - pt) ** self.gamma)
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=3.0, focal_weight=0.8, dice_weight=0.2):
        super().__init__()
        self.focal_loss = FocalLoss(gamma)
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        # Focal Loss con peso de clase
        focal = self.focal_loss(pred, target)
        focal_weighted = torch.where(target == 1, 
                                   self.alpha * focal, 
                                   (1 - self.alpha) * focal)
        
        # Dice Loss
        dice = self.dice_loss(pred, target)
        
        # Pérdida combinada
        combined_loss = self.focal_weight * focal_weighted.mean() + \
                       self.dice_weight * dice
        
        return combined_loss

def train_evaluate_fold(model, X_train, y_train, X_val, y_val, criterion, optimizer, 
                       batch_size, n_epochs, patience, device='cpu'):
    """Entrena y evalúa el modelo en un fold"""
    # Asegurar que y_train y y_val tengan la forma correcta
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    
    # Crear dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Variables para early stopping
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Modo entrenamiento
        model.train()
        total_loss = 0
        train_preds = []
        train_true = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Pérdida del criterio (la regularización L2 ya está en el optimizador)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy() > 0.5)
            train_true.extend(batch_y.cpu().numpy())
        
        # Calcular métricas de entrenamiento
        train_f1 = f1_score(train_true, train_preds)
        avg_loss = total_loss / len(train_loader)
        
        # Modo evaluación
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_preds.extend(outputs.cpu().numpy() > 0.5)
                val_true.extend(batch_y.cpu().numpy())
        
        # Calcular F1-score en validación
        val_f1 = f1_score(val_true, val_preds)
        
        # Imprimir progreso
        print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_loss:.4f} - Train F1: {train_f1:.4f} - Val F1: {val_f1:.4f}')
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping en epoch {epoch+1}')
                break
    
    return best_val_f1
    # Graficar curvas de aprendizaje
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Validation F1')
    plt.title('F1-Score durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    
    plt.tight_layout()
    
    # Guardar gráfico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'models/learning_curves_{timestamp}.png')
    plt.close()
    
    return best_val_f1, best_model_state

def find_optimal_threshold(model, X_val, y_val, n_thresholds=100):
    """Encuentra el umbral óptimo para maximizar el F1-score"""
    model.eval()
    with torch.no_grad():
        val_outputs = model(torch.FloatTensor(X_val)).cpu().numpy()
    
    best_threshold = 0.5
    best_f1 = 0
    
    thresholds = np.linspace(0.1, 0.9, n_thresholds)
    for threshold in thresholds:
        val_preds = (val_outputs >= threshold).astype(float)
        f1 = f1_score(y_val, val_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

def objective(trial, X_train, y_train, X_val, y_val, device):
    """Función objetivo para Optuna"""
    # Hiperparámetros a optimizar
    params = {
        # Parámetros del modelo
        'n_layers': trial.suggest_int('n_layers', 2, 4),
        'layer_size': trial.suggest_int('layer_size', 32, 128),
        'dropout': trial.suggest_float('dropout', 0.3, 0.7),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        
        # Parámetros de la función de pérdida
        'focal_weight': trial.suggest_float('focal_weight', 0.6, 0.9),
        'dice_weight': trial.suggest_float('dice_weight', 0.1, 0.4),
        'focal_gamma': trial.suggest_float('focal_gamma', 2.0, 4.0),
        'focal_alpha': trial.suggest_float('focal_alpha', 0.25, 0.45)
    }
    
    # Inicializar modelo y moverlo al dispositivo correcto
    model = ImprovedNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=params['layer_size'],
        n_layers=params['n_layers'],
        dropout=params['dropout']
    ).to(device)
    
    # Configurar función de pérdida y optimizador
    criterion = CombinedLoss(
        alpha=params['focal_alpha'],
        gamma=params['focal_gamma'],
        focal_weight=params['focal_weight'],
        dice_weight=params['dice_weight']
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    # Entrenar y evaluar
    val_f1 = train_evaluate_fold(
        model, X_train, y_train, X_val, y_val,
        criterion, optimizer, params['batch_size'],
        n_epochs=100, patience=10
    )
    
    return val_f1

def optimize_hyperparameters(X_train, y_train, X_val, y_val, device='cpu', n_trials=50):
    """Optimiza hiperparámetros usando Optuna"""
    logging.info('Iniciando optimización de hiperparámetros...')
    
    # Crear estudio
    study = optuna.create_study(direction='maximize')
    
    # Crear función objetivo con los datos
    objective_with_data = lambda trial: objective(trial, X_train, y_train, X_val, y_val, device)
    
    # Optimizar
    study.optimize(objective_with_data, n_trials=n_trials)
    
    logging.info('\nOptimización completada exitosamente')
    logging.info('\nMejores hiperparámetros encontrados:')
    for key, value in study.best_params.items():
        logging.info(f"\n{key}: {value}")
    
    logging.info(f"\n\nMejor F1-Score promedio: {study.best_value:.4f}")
    
    # Si no hay timestamp, crear uno
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Asegurar que el directorio models existe
    os.makedirs('models', exist_ok=True)
    
    # Copiar todos los mejores parámetros
    best_params = study.best_params.copy()
    optimization_results = {
        'best_params': best_params,
        'best_value': float(study.best_value),
        'timestamp': timestamp
    }
    
    results_file = f'models/optimization_results_{timestamp}.json'
    logging.info(f'Guardando resultados en: {results_file}')
    with open(results_file, 'w') as f:
        json.dump(optimization_results, f, indent=4)
        
    return best_params

def train_final_model(X_train, y_train, X_val, y_val, best_params, device='cpu'):
    """Entrena el modelo final con los mejores hiperparámetros y evalúa su rendimiento"""
    logging.info('Iniciando entrenamiento del modelo final...')
    
    try:
        # Crear modelo con los mejores hiperparámetros
        model = ImprovedNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_size=best_params['layer_size'],
            n_layers=best_params['n_layers'],
            dropout=best_params['dropout']
        ).to(device)
        
        # Crear función de pérdida y optimizador
        criterion = CombinedLoss(
            alpha=best_params['focal_alpha'],
            gamma=best_params['focal_gamma'],
            focal_weight=best_params['focal_weight'],
            dice_weight=best_params['dice_weight']
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay']
        )
        
        # Preparar datos de entrenamiento y validación
        train_dataset = TensorDataset(X_train, y_train.reshape(-1, 1))
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        
        val_dataset = TensorDataset(X_val, y_val.reshape(-1, 1))
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        # Listas para almacenar métricas
        train_losses = []
        val_losses = []
        val_f1_scores = []
        
        # Entrenamiento
        n_epochs = 100
        early_stopping = EarlyStopping(patience=10)
        best_val_f1 = 0.0
        
        for epoch in range(n_epochs):
            # Modo entrenamiento
            model.train()
            total_train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Modo evaluación
            model.eval()
            total_val_loss = 0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    val_loss = criterion(outputs, batch_y)
                    total_val_loss += val_loss.item()
                    
                    # Guardar predicciones y valores reales
                    val_preds.extend((outputs >= 0.5).float().cpu().numpy())
                    val_true.extend(batch_y.cpu().numpy())
            
            # Calcular métricas
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            val_f1 = f1_score(val_true, val_preds)
            val_f1_scores.append(val_f1)
            
            # Actualizar mejor F1-score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()
            
            # Early stopping
            early_stopping.step(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                logging.info(f'[Época {epoch + 1}] '
                          f'Train Loss: {avg_train_loss:.4f}, '
                          f'Val Loss: {avg_val_loss:.4f}, '
                          f'Val F1: {val_f1:.4f}')
            
            if early_stopping.early_stop:
                logging.info(f'Early stopping en época {epoch + 1}')
                break
        
        logging.info('Entrenamiento completado')
        logging.info(f'Mejor F1-Score en validación: {best_val_f1:.4f}')
        
        # Restaurar mejor modelo
        model.load_state_dict(best_model_state)
        
        # Guardar modelo y métricas
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join('models', f'model_{timestamp}.pt')
        logging.info(f'Guardando modelo en: {model_path}')
        
        save_dict = {
            'state_dict': model.state_dict(),
            'hyperparameters': {
                'input_size': X_train.shape[1],
                'hidden_size': best_params['layer_size'],
                'n_layers': best_params['n_layers'],
                'dropout': best_params['dropout']
            },
            'metrics': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_f1_scores': val_f1_scores,
                'best_val_f1': best_val_f1
            }
        }
        
        torch.save(save_dict, model_path)
        
        # Guardar gráficas de entrenamiento
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Learning Curves - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(val_f1_scores, label='Validation F1')
        plt.title('Learning Curve - F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join('models', f'learning_curves_{timestamp}.png'))
        plt.close()
        
        return True
        
    except Exception as e:
        logging.error(f'Error en el entrenamiento del modelo final: {str(e)}')
        return False
        # Guardar modelo y hiperparámetros
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join('models', f'model_{timestamp}.pt')
        logging.info(f'Guardando modelo en: {model_path}')
        
        # Guardar estado del modelo y hiperparámetros juntos
        save_dict = {
            'state_dict': model.state_dict(),
            'hyperparameters': {
                'input_size': X_train.shape[1],
                'hidden_size': best_params['layer_size'],
                'n_layers': best_params['n_layers'],
                'dropout': best_params['dropout']
            }
        }
        torch.save(save_dict, model_path)
        
        return True
        
    except Exception as e:
        logging.error(f'Error en el entrenamiento del modelo final: {str(e)}')
        return False
        
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Early stopping
            if early_stopping.step(avg_loss):
                logging.info(f'Early stopping en época {epoch + 1}')
                break
            
            if (epoch + 1) % 10 == 0:
                logging.info(f'[Época {epoch + 1}] Pérdida: {avg_loss:.4f}')
        
        logging.info('Entrenamiento completado')
        return True
        
    except Exception as e:
        logging.error(f'Error en entrenamiento: {e}')
        return False
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Variables para early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience = 15  # Aumentado para el modelo final
        
        # Listas para almacenar métricas
        train_losses = []
        train_f1s = []
        val_f1s = []
        
        # Entrenamiento
        n_epochs = 200  # Más épocas para el modelo final
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
            train_preds = []
            train_true = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                # Pérdida del criterio + regularización L2
                loss = criterion(outputs, batch_y)
                loss += model.get_l2_regularization()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                train_preds.extend((outputs >= 0.5).float().detach().numpy())
                train_true.extend(batch_y.numpy())
            
            # Calcular métricas de entrenamiento
            avg_loss = total_loss / len(train_loader)
            train_f1 = f1_score(train_true, train_preds)
            
            # Calcular F1 en validación
            model.eval()
            with torch.no_grad():
                val_outputs = model(torch.FloatTensor(X_val))
                val_preds = (val_outputs >= 0.5).float()
                val_f1 = f1_score(y_val, val_preds)
            
            train_losses.append(avg_loss)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
            
            print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_loss:.4f} - Train F1: {train_f1:.4f} - Val F1: {val_f1:.4f}')
            
            # Early stopping basado en pérdida
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Guardar mejor modelo
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hyperparameters': {
                        **best_params,
                        'input_size': X_train.shape[1]
                    },
                    'train_losses': train_losses,
                    'train_f1s': train_f1s,
                    'val_f1s': val_f1s
                }, os.path.join('models', 'model_final.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping en epoch {epoch+1}')
                    break
        
        # Encontrar umbral óptimo
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val))
        
        best_threshold = 0.5
        best_val_f1 = 0
        
        for threshold in np.arange(0.1, 0.9, 0.01):
            val_preds = (val_outputs >= threshold).float()
            val_f1 = f1_score(y_val, val_preds)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_threshold = threshold
        
        print(f'\nRendimiento del modelo final:')
        print(f'Umbral óptimo: {best_threshold:.3f}')
        
        # Evaluar en conjunto de prueba
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_preds = (test_outputs >= best_threshold).float()
            test_f1 = f1_score(y_test, test_preds)
        
        print(f'F1-Score en validación: {best_val_f1:.4f}')
        print(f'F1-Score en prueba: {test_f1:.4f}')
        
        # Guardar umbral óptimo junto con el modelo
        checkpoint = torch.load(os.path.join('models', f'model_{timestamp}.pt'))
        checkpoint['optimal_threshold'] = best_threshold
        torch.save(checkpoint, os.path.join('models', f'model_{timestamp}.pt'))
        
        print(f'\nModelo guardado en: models/model_{timestamp}.pt')
        
        # Graficar curvas de aprendizaje
        plt.figure(figsize=(15, 5))
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Gráfico de F1-Score
        plt.subplot(1, 2, 2)
        plt.plot(train_f1s, label='Train F1')
        plt.plot(val_f1s, label='Validation F1')
        plt.title('F1-Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1-Score')
        plt.legend()
        
        # Guardar gráfico
        plt.savefig(os.path.join('data', 'final', f'learning_curves_{timestamp}.png'))
        plt.close()
        
        print(f'F1-Score en validación: {best_val_f1:.4f}')
        print(f'F1-Score en prueba: {test_f1:.4f}\n')
        
        # Actualizar el modelo guardado con el umbral óptimo
        model_path = os.path.join('models', 'model_final.pt')
        checkpoint = torch.load(model_path)
        checkpoint['optimal_threshold'] = best_threshold
        torch.save(checkpoint, model_path)
        print(f'Modelo actualizado con umbral óptimo en: {model_path}\n')
        
        return True
        
    except Exception as e:
        print(f'Error en el entrenamiento del modelo final: {str(e)}')
        return False

def prepare_data(mode='train'):
    """Prepara los datos para entrenamiento o inferencia
    Args:
        timestamp: Timestamp para cargar los archivos correctos
        mode: 'train' para entrenamiento, 'inference' para inferencia
    """
    try:
        # Configurar dispositivo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar datos procesados
        logging.info("Cargando datos procesados...")
        
        train_file = os.path.join('.', 'data', 'processed', 'train_processed.csv')
        test_file = os.path.join('.', 'data', 'processed', 'test_processed.csv')
        
        # En modo entrenamiento solo necesitamos el archivo de train
        if mode == 'train':
            if not os.path.exists(train_file):
                logging.error(f"No se encontró el archivo de entrenamiento")
                return None, None, None, None, None, None
                
            train_data = pd.read_csv(train_file)
            
            # Limpiar nombres de columnas
            train_data.columns = train_data.columns.str.strip()
            
            # Separar features y target
            X = train_data.drop('target', axis=1)
            y = train_data['target']
            
            # Split en train y validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Convertir a tensores y mover al dispositivo correcto
            X_train_tensor = torch.FloatTensor(X_train.values).to(device)
            y_train_tensor = torch.FloatTensor(y_train.values).to(device)
            X_val_tensor = torch.FloatTensor(X_val.values).to(device)
            y_val_tensor = torch.FloatTensor(y_val.values).to(device)
            
            return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, None, None
            
        # En modo inferencia cargamos el archivo de test
        else:
            if not os.path.exists(test_file):
                logging.error(f"No se encontró el archivo de test")
                return None, None, None, None, None, None
                
            test_data = pd.read_csv(test_file)
            test_data.columns = test_data.columns.str.strip()
            X_test_tensor = torch.FloatTensor(test_data.values).to(device)
            
            return None, None, None, None, X_test_tensor, None
        
        # Guardar pipeline
        preprocessor_file = os.path.join('.', 'models', 'preprocessor.joblib')
        joblib.dump(preprocessing_pipeline, preprocessor_file)
        logging.info(f'Pipeline guardado en: {preprocessor_file}')
        
        return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, preprocessing_pipeline
        
    except Exception as e:
        logging.error(f'Error en preparación de datos: {e}')
        raise

def main():
    """Función principal"""
    try:
        # Obtener argumentos
        train_final = False
        best_params = None
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == '--train-final':
                train_final = True
                i += 1
            elif sys.argv[i] == '--best-params' and i + 1 < len(sys.argv):
                best_params = json.loads(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        
        # Cargar y preparar datos
        logging.info("Cargando y preparando datos...")
        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, preprocessing_pipeline = prepare_data(mode='train')
        
        # Configurar dispositivo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if train_final and best_params:
            # Entrenar modelo final con los hiperparámetros proporcionados
            logging.info("Entrenando modelo final con hiperparámetros proporcionados...")
            success = train_final_model(
                X_train_tensor, y_train_tensor,
                X_val_tensor, y_val_tensor,
                best_params,
                device=device
            )
            
            if success:
                logging.info("Modelo final entrenado exitosamente")
                return 0
            else:
                logging.error("Error en entrenamiento del modelo final")
                return 1
        else:
            # Optimizar hiperparámetros
            logging.info("Iniciando optimización de hiperparámetros...")
            best_params = optimize_hyperparameters(
                X_train_tensor, y_train_tensor,
                X_val_tensor, y_val_tensor,
                device=device
            )
            return 0
            
    except Exception as e:
        logging.error(f"Error en ejecución: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
