import pandas as pd
import numpy as np
import os
import sys
import torch
import json
import logging
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
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
)

# Definir rutas
path_processed = os.path.join('data', 'processed')
path_models = os.path.join('models')

# Asegurar que los directorios existan
os.makedirs(path_processed, exist_ok=True)
os.makedirs(path_models, exist_ok=True)

def train_final_model(X_train, y_train, X_val, y_val, params, device='cpu', seed=42):
    """Entrena el modelo final con los hiperparámetros optimizados"""
    # Fijar semillas para reproducibilidad
    set_seed(seed)
    try:
        # Validar dimensiones de entrada
        logging.info(f"Dimensiones de entrada: X_train={X_train.shape}, y_train={y_train.shape}")
        logging.info(f"Dimensiones de validación: X_val={X_val.shape}, y_val={y_val.shape}")
        
        # Convertir a tensores
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
        
        # Mover tensores al dispositivo
        X_train_tensor = X_train_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        X_val_tensor = X_val_tensor.to(device)
        y_val_tensor = y_val_tensor.to(device)
        
        # Crear modelo con los hiperparámetros proporcionados
        input_size = X_train_tensor.shape[1]
        model = ImprovedNeuralNetwork(
            input_size=input_size,
            hidden_size=params['layer_size'],
            n_layers=params['n_layers'],
            dropout=params['dropout']
        ).to(device)
        
        # Crear datasets y dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
        
        # Configurar optimizador y función de pérdida
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        criterion = CombinedLoss(
            focal_weight=params['focal_weight'],
            gamma=params['focal_gamma'],
            alpha=params['focal_alpha'],
            dice_weight=params['dice_weight']
        )
        
        # Variables para early stopping
        early_stopping = EarlyStopping(patience=10)
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_val_f1 = 0
        
        # Listas para almacenar métricas
        train_losses = []
        val_losses = []
        train_f1_scores = []
        val_f1_scores = []
        
        # Entrenamiento
        n_epochs = 100
        for epoch in range(n_epochs):
            # Modo entrenamiento
            model.train()
            train_loss = 0.0
            train_outputs = []
            train_true = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                train_outputs.extend(outputs.detach().cpu().numpy())
                train_true.extend(batch_y.cpu().numpy())
            
            # Modo evaluación
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_true = []
            
            val_outputs = []
            val_true = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    val_outputs.extend(outputs.cpu().numpy())
                    val_true.extend(batch_y.cpu().numpy())
            
            # Calcular pérdidas promedio
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Optimizar umbral usando validación
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_threshold = 0.5
            best_val_f1 = 0
            
            for threshold in thresholds:
                val_preds = (np.array(val_outputs) >= threshold).astype(float)
                current_f1 = f1_score(val_true, val_preds)
                
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_threshold = threshold
            
            # Calcular F1-scores con el mejor umbral
            train_preds = (np.array(train_outputs) >= best_threshold).astype(float)
            val_preds = (np.array(val_outputs) >= best_threshold).astype(float)
            
            train_f1 = f1_score(train_true, train_preds)
            val_f1 = f1_score(val_true, val_preds)
            
            # Almacenar métricas
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_f1_scores.append(train_f1)
            val_f1_scores.append(val_f1)
            
            # Logging
            logging.info(f"Epoch {epoch+1}/{n_epochs}:")
            logging.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            logging.info(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
            

            
            # Early stopping y guardado del mejor modelo
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
                best_val_f1 = val_f1
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping en epoch {epoch+1}")
                    break
        
        # Generar gráficos de curvas de aprendizaje
        plt.figure(figsize=(12, 4))
        
        # Subplot para pérdidas
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Learning Curves - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Subplot para F1-scores
        plt.subplot(1, 2, 2)
        plt.plot(train_f1_scores, label='Train F1')
        plt.plot(val_f1_scores, label='Val F1')
        plt.title('Learning Curves - F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.tight_layout()
        curves_path = os.path.join(path_models, 'learning_curves.png')
        plt.savefig(curves_path)
        plt.close()
        
        # Guardar el mejor modelo y métricas
        model.load_state_dict(best_model_state)
        model_path = os.path.join(path_models, 'model_final.pt')
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'hyperparameters': {
                'input_size': input_size,
                'hidden_size': params['layer_size'],
                'n_layers': params['n_layers'],
                'dropout': params['dropout'],
                'optimal_threshold': best_threshold
            },
            'metrics': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_f1_scores': val_f1_scores,
                'best_val_f1': best_val_f1
            }
        }
        
        torch.save(save_dict, model_path)
        logging.info(f'Modelo guardado en: {model_path}')
        logging.info(f'Learning curves guardadas en: {curves_path}')
        logging.info(f'Mejor F1-Score en validación: {best_val_f1:.4f}')
        
        return True
    except Exception as e:
        logging.error(f"Error en entrenamiento: {str(e)}")
        return False

def main():
    """Función principal para entrenar el modelo final"""
    # Fijar semillas para reproducibilidad
    set_seed(42)
    try:
        # Cargar hiperparámetros del archivo de optimización
        optimization_file = os.path.join('models', 'optimization_results.json')
        if not os.path.exists(optimization_file):
            logging.error(f"No se encontró el archivo de optimización: {optimization_file}")
            return 1
            
        with open(optimization_file, 'r') as f:
            optimization_results = json.load(f)
            best_params = optimization_results['best_params']
        
        logging.info("Hiperparámetros cargados:")
        for param, value in best_params.items():
            logging.info(f"{param}: {value}")
        
        # Cargar datos procesados
        logging.info("\nCargando datos procesados...")
        train_file = os.path.join(path_processed, 'train_processed.csv')
        
        if not os.path.exists(train_file):
            logging.error(f"El archivo {train_file} no existe")
            return 1
        
        try:
            data = pd.read_csv(train_file)
            
            # Separar features y target
            if 'target' not in data.columns:
                logging.error("La columna 'target' no está presente en los datos")
                return 1
                
            y = data['target'].values
            X = data.drop('target', axis=1).values
            
            # Validar datos antes del split
            logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
            logging.info(f"Valores únicos en y: {dict(zip(*np.unique(y, return_counts=True)))}")
            
            if len(X) != len(y):
                logging.error(f"Número inconsistente de muestras: X={len(X)}, y={len(y)}")
                return 1
            
            # Split train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logging.info(f"Split completado:")
            logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            logging.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
            
        except Exception as e:
            logging.error(f"Error al cargar o procesar los datos: {str(e)}")
            return 1
        
        # Configurar dispositivo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Usando dispositivo: {device}")
        
        # Entrenar modelo final
        logging.info("\nIniciando entrenamiento del modelo final...")
        try:
            success = train_final_model(
                X_train, y_train,
                X_val, y_val,
                best_params,
                device=device
            )
            
            if success:
                logging.info("Entrenamiento completado exitosamente")
                return 0
            else:
                logging.error("Error en entrenamiento del modelo final")
                return 1
        except Exception as e:
            logging.error(f"Error en entrenamiento: {str(e)}")
            return 1
            
    except Exception as e:
        logging.error(f"Error en main: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
