import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
from datetime import datetime

# Encontrar el archivo de resultados más reciente
path_models = os.path.join('.', 'models')
optimization_files = [f for f in os.listdir(path_models) if f.startswith('optimization_results_')]
latest_file = sorted(optimization_files)[-1]

# Cargar resultados
with open(os.path.join(path_models, latest_file), 'r') as f:
    optimization_results = json.load(f)

# Crear un DataFrame con los mejores parámetros
best_params = pd.DataFrame([optimization_results['best_parameters']])
best_f1 = optimization_results['best_f1_score']

# Crear visualizaciones
plt.style.use('seaborn')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análisis de los Mejores Hiperparámetros', fontsize=16)

# 1. Arquitectura de la Red
ax = axes[0, 0]
layer_sizes = [best_params['layer1_size'].iloc[0], 
               best_params['layer2_size'].iloc[0], 
               best_params['layer3_size'].iloc[0]]
ax.bar(['Capa 1', 'Capa 2', 'Capa 3'], layer_sizes)
ax.set_title('Arquitectura de la Red Neural')
ax.set_ylabel('Número de Neuronas')

# 2. Parámetros de Regularización
ax = axes[0, 1]
reg_params = [best_params['dropout_rate'].iloc[0], 
              best_params['weight_decay'].iloc[0]]
ax.bar(['Dropout Rate', 'Weight Decay'], reg_params)
ax.set_title('Parámetros de Regularización')

# 3. Parámetros de Entrenamiento
ax = axes[1, 0]
train_params = [best_params['learning_rate'].iloc[0],
                best_params['batch_size'].iloc[0]/1000]
ax.bar(['Learning Rate', 'Batch Size (÷1000)'], train_params)
ax.set_title('Parámetros de Entrenamiento')

# 4. Parámetros de la Función de Pérdida
ax = axes[1, 1]
loss_params = [best_params['focal_gamma'].iloc[0],
               best_params['loss_alpha'].iloc[0]]
ax.bar(['Focal Gamma', 'Loss Alpha'], loss_params)
ax.set_title('Parámetros de la Función de Pérdida')

plt.tight_layout()
plt.savefig(os.path.join(path_models, 'hyperparameters_analysis.png'))
print(f"\nAnálisis guardado en: {os.path.join(path_models, 'hyperparameters_analysis.png')}")

# Análisis detallado de los hiperparámetros
print("\nAnálisis de los Mejores Hiperparámetros:")

print("\n1. Arquitectura de la Red:")
print(f"  - Primera capa: {best_params['layer1_size'].iloc[0]} neuronas")
print(f"  - Segunda capa: {best_params['layer2_size'].iloc[0]} neuronas")
print(f"  - Tercera capa: {best_params['layer3_size'].iloc[0]} neuronas")
print(f"  - Función de activación: {best_params['activation'].iloc[0]}")

print("\n2. Regularización:")
print(f"  - Dropout rate: {best_params['dropout_rate'].iloc[0]:.4f}")
print(f"  - Weight decay: {best_params['weight_decay'].iloc[0]:.6f}")

print("\n3. Entrenamiento:")
print(f"  - Learning rate: {best_params['learning_rate'].iloc[0]:.6f}")
print(f"  - Batch size: {best_params['batch_size'].iloc[0]}")

print("\n4. Función de Pérdida:")
print(f"  - Focal Loss gamma: {best_params['focal_gamma'].iloc[0]:.4f}")
print(f"  - Loss alpha: {best_params['loss_alpha'].iloc[0]:.4f}")

print(f"\nMejor F1-Score alcanzado: {best_f1:.4f}")
