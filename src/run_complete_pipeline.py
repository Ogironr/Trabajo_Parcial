import os
import json
import logging
import subprocess
from datetime import datetime

def setup_logging():
    """Configura el logging"""
    log_file = 'pipeline.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_directories():
    """Crea la estructura de directorios necesaria"""
    directories = ['./data/raw', './data/processed', './data/final', './models', './logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f'Directorio creado/verificado: {directory}')

def run_hyperparameter_optimization():
    """Ejecuta la optimización de hiperparámetros"""
    logging.info('Iniciando optimización de hiperparámetros...')
    try:
        # Ejecutar optimización
        logging.info('Ejecutando optimización de hiperparámetros...')
        optimization_file = os.path.join('models', 'optimization_results.json')
        
        # Si existe el archivo previo, lo eliminamos para forzar nueva optimización
        if os.path.exists(optimization_file):
            os.remove(optimization_file)
            logging.info('Archivo de resultados previo eliminado para nueva optimización')
        
        # Si no existe, ejecutar optimización
        logging.info('Ejecutando nueva optimización de hiperparámetros...')
        result = subprocess.run(
            ['python', 'src/og_mvp_windsurf_tuned.py'],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Verificar si se creó el archivo de resultados
        if os.path.exists(optimization_file):
            with open(optimization_file, 'r') as f:
                results = json.load(f)
                if 'best_params' in results:
                    logging.info('Hiperparámetros óptimos encontrados')
                    return results['best_params']
                else:
                    logging.error('No se encontraron hiperparámetros óptimos en el archivo')
                    return None
        else:
            logging.error('No se encontró el archivo de resultados')
            return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Error en la optimización: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Error inesperado: {str(e)}")
        return False

def train_final_model(best_params):
    """Entrena el modelo final con los mejores hiperparámetros"""
    logging.info('Iniciando entrenamiento del modelo final...')
    try:
        # Ejecutar entrenamiento con los mejores hiperparámetros
        result = subprocess.run(
            ['python', 'src/og_train_final_model.py', '--best-params', json.dumps(best_params)],
            check=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logging.info('Entrenamiento completado exitosamente')
            return True
        else:
            logging.error(f'Error en entrenamiento:\n{result.stderr}')
            return False
    except subprocess.CalledProcessError as e:
        logging.error(f'\nError en entrenamiento: {e}')
        return False
    except Exception as e:
        logging.error(f'\nError inesperado en entrenamiento: {e}')
        return False
        return False

def run_inference():
    """Ejecuta la inferencia con el modelo entrenado"""
    logging.info('Iniciando inferencia...')
    try:
        # Ejecutar inferencia
        result = subprocess.run(
            ['python', 'src/og_inferencia_windsurf_optimized.py'],
            check=True
        )
        
        # Verificar que se generaron las predicciones
        predictions_file = os.path.join('data/final', 'submission_final.csv')
        if os.path.exists(predictions_file):
            logging.info('Predicciones generadas exitosamente')
            return True
        
        logging.error('No se encontró el archivo de predicciones')
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f'\nError en inferencia: {e}')
        if e.output:
            logging.error(f'Salida de error:\n{e.output}')
        return False
    except Exception as e:
        logging.error(f'\nError inesperado: {e}')
        return False

def analyze_results():
    """Analiza los resultados del entrenamiento"""
    logging.info('Iniciando análisis de resultados...')
    try:
        # Verificar que existan los archivos necesarios
        model_file = os.path.join('models', 'model_final.pt')
        if not os.path.exists(model_file):
            logging.error(f'\nNo se encontró el archivo del modelo: {model_file}')
            return False
            
        # Ejecutar análisis
        result = subprocess.run(
            ['python', 'src/og_analisis_completo.py'],
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logging.info('Salida de análisis:\n' + result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f'\nError en análisis: {e}')
        if e.output:
            logging.error(f'Salida de error:\n{e.output}')
        return False
    except Exception as e:
        logging.error(f'\nError inesperado en análisis: {e}')
        return False

def save_pipeline_metadata(success):
    """Guarda los metadatos de la ejecución del pipeline"""
    metadata = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'success': success
    }
    
    metadata_file = os.path.join('logs', 'pipeline_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logging.info(f'Metadatos del pipeline guardados en {metadata_file}')

def process_data():
    """Ejecuta el procesamiento de datos"""
    logging.info('Iniciando procesamiento de datos...')
    try:
        # Ejecutar procesamiento con el timestamp
        logging.info('Ejecutando script de procesamiento...')
        result = subprocess.run(
            ['python', 'src/og_procesamiento.py'],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Mostrar la salida del script
        if result.stdout:
            logging.info('Salida del procesamiento:\n' + result.stdout)
        
        # Verificar que se generaron los archivos procesados
        processed_file = os.path.join('data/processed', 'train_processed.csv')
        if os.path.exists(processed_file):
            logging.info(f'Datos procesados generados exitosamente en {processed_file}')
            return True
        
        logging.error(f'No se encontró el archivo de datos procesados: {processed_file}')
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error en el procesamiento: {str(e)}")
        if e.output:
            logging.error(f'Salida de error:\n{e.output}')
        return False
    except Exception as e:
        logging.error(f"Error inesperado en procesamiento: {str(e)}")
        return False

def main():
    """Ejecuta el pipeline completo"""
    setup_logging()
    logging.info('Iniciando pipeline completo...')
    
    # Crear estructura de directorios
    create_directories()
    
    # Ejecutar cada paso del pipeline
    success = True
    try:
        # 1. Procesamiento de datos
        logging.info('Iniciando procesamiento de datos...')
        proc_result = process_data()
        if not proc_result:
            raise Exception("Error en procesamiento de datos")
        logging.info('Procesamiento completado exitosamente')
        
        # 2. Optimización de hiperparámetros
        logging.info('Iniciando optimización de hiperparámetros...')
        best_params = run_hyperparameter_optimization()
        if best_params is None:
            raise Exception("Error en optimización de hiperparámetros")
        logging.info('Optimización completada exitosamente')
        logging.info(f'Mejores hiperparámetros encontrados: {best_params}')
        
        # 3. Entrenamiento del modelo final
        logging.info('Iniciando entrenamiento del modelo final...')
        train_result = train_final_model(best_params)
        if not train_result:
            raise Exception("Error en entrenamiento del modelo final")
        logging.info('Entrenamiento completado exitosamente')
        
        # 4. Inferencia
        logging.info('Iniciando inferencia con el modelo entrenado...')
        infer_result = run_inference()
        if not infer_result:
            raise Exception("Error en inferencia")
        logging.info('Inferencia completada exitosamente')
        
        # 5. Análisis de resultados
        logging.info('Iniciando análisis de resultados...')
        analysis_result = analyze_results()
        if not analysis_result:
            raise Exception("Error en análisis de resultados")
        logging.info('Análisis completado exitosamente')
            
    except Exception as e:
        logging.error(f'Error en el pipeline: {e}')
        success = False
    
    # Guardar metadatos del pipeline
    save_pipeline_metadata(success)
    
    if not success:
        logging.error('Pipeline completado con errores')
    else:
        logging.info('Pipeline completado exitosamente')

if __name__ == "__main__":
    main()
