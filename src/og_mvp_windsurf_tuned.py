import pandas as pd
import numpy as np
import os
import sys
import torch
import optuna
import logging
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import json
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from models import ImprovedNeuralNetwork, EarlyStopping, CombinedLoss

def set_seed(seed=42):
    """Fijar semillas para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)  # Definir rutas
path_processed = os.path.join('.', 'data', 'processed')
path_models = os.path.join('.', 'models')

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

def objective(trial, X_train, y_train, X_val, y_val, device='cpu', seed=42):
    # Fijar semillas para reproducibilidad
    set_seed(seed)
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

def optimize_hyperparameters(X_train, y_train, X_val, y_val, device='cpu', n_trials=50, seed=42):
    # Fijar semillas para reproducibilidad
    set_seed(seed)
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
    
    # Obtener mejores hiperparámetros
    best_params = study.best_params
    best_value = study.best_value
    
    # Imprimir resultados
    for param, value in best_params.items():
        logging.info(f'{param}: {value}')
    
    logging.info(f'\nMejor F1-Score promedio: {best_value:.4f}')
    
    # Guardar resultados
    results_file = os.path.join('models', 'optimization_results.json')
    results = {
        'best_params': best_params,
        'best_value': best_value
    }
    
    os.makedirs('models', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f'Resultados guardados en: {results_file}')
        
    return best_params

def main():
    """Función principal para optimización de hiperparámetros"""
    try:
        # Preparar datos
        logging.info("Preparando datos...")
        train_file = os.path.join('data/processed', 'train_processed.csv')
        
        if not os.path.exists(train_file):
            logging.error("No se encontró el archivo de entrenamiento")
            return 1
            
        train_data = pd.read_csv(train_file)
        
        # Verificar columnas
        logging.info(f"Columnas disponibles: {len(train_data.columns)} columnas")
        
        # Separar features y target
        X = train_data.drop(['target'], axis=1)
        y = train_data['target']
        
        # Convertir a float32 para reducir uso de memoria
        X = X.astype('float32')
        y = y.astype('float32')
        
        # Dividir en train y validación
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        # Convertir a tensores
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train_tensor = torch.FloatTensor(X_train.values).to(device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(device)
        X_val_tensor = torch.FloatTensor(X_val.values).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values).to(device)
        
        # Optimizar hiperparámetros
        logging.info("Iniciando optimización de hiperparámetros...")
        best_params = optimize_hyperparameters(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, device=device)
        
        if best_params is None:
            logging.error("Error en la optimización")
            return 1
        
        return 0
            
    except Exception as e:
        logging.error(f"Error en ejecución: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
