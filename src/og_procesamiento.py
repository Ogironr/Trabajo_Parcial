import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy import stats
import joblib

def create_directories():
    """Crea la estructura de directorios necesaria"""
    directories = ['./data/raw', './data/processed', './data/final', './models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f'Directorio creado/verificado: {directory}')

def load_data(filename, sep=';'):
    """Carga los datos desde un archivo CSV"""
    filepath = os.path.join('.', 'data', 'raw', filename)
    df = pd.read_csv(filepath, sep=sep)
    
    # Asegurar que la columna target se llame 'ZSN'
    if 'target' in df.columns:
        df.rename(columns={'target': 'ZSN'}, inplace=True)
    
    return df

def identify_column_types(df):
    """Identifica los tipos de columnas en el dataset"""
    # Identificar columnas numéricas y categóricas
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_features = [col for col in numeric_features if col not in ['ID', 'ZSN']]
    
    categorical_features = df.select_dtypes(include=['object']).columns
    categorical_features = [col for col in categorical_features if col not in ['ID', 'ZSN']]
    
    return numeric_features, categorical_features

def create_feature_engineering_pipeline(numeric_features, categorical_features):
    """Crea el pipeline de preprocesamiento con feature engineering"""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def create_domain_features(df):
    """Crea características basadas en el dominio"""
    # Asegurarnos de que el DataFrame sea una copia para no modificar el original
    df = df.copy()
    
    # Problemas pulmonares
    pulmonary_cols = ['zab_leg_01', 'zab_leg_02', 'zab_leg_03', 'zab_leg_04', 'zab_leg_06']
    available_pulmonary = [col for col in pulmonary_cols if col in df.columns]
    if available_pulmonary:
        df['PULMONARY_ISSUES_COUNT'] = df[available_pulmonary].sum(axis=1)
    else:
        df['PULMONARY_ISSUES_COUNT'] = pd.Series(0, index=df.index)

    # Problemas endocrinos (diabetes)
    diabetes_cols = ['endocr_01', 'endocr_02', 'endocr_03']
    available_diabetes = [col for col in diabetes_cols if col in df.columns]
    if available_diabetes:
        df['DIABETES_ISSUES_COUNT'] = df[available_diabetes].sum(axis=1)
    else:
        df['DIABETES_ISSUES_COUNT'] = pd.Series(0, index=df.index)

    # Problemas neurológicos
    neuro_cols = [
        'nr_11', 'nr_01', 'nr_02', 'nr_03', 'nr_04', 'nr_07', 'nr_08',
        'np_01', 'np_04', 'np_05', 'np_07', 'np_08', 'np_09', 'np_10'
    ]
    available_neuro = [col for col in neuro_cols if col in df.columns]
    if available_neuro:
        df['NEURO_ISSUES_COUNT'] = df[available_neuro].sum(axis=1)
    else:
        df['NEURO_ISSUES_COUNT'] = pd.Series(0, index=df.index)

    return df

def process_dataset(df, preprocessor=None, train_mode=True):
    """Procesa el dataset aplicando feature engineering y preprocesamiento"""
    # Guardar el target si existe
    target = None
    if 'ZSN' in df.columns:
        target = df['ZSN'].copy()
        df = df.drop('ZSN', axis=1)
    
    # Eliminar columna ID si existe
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Crear características de dominio
    df = create_domain_features(df)
    
    # Identificar tipos de columnas
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # En modo entrenamiento, crear nuevo preprocessor
    if train_mode:
        preprocessor = create_feature_engineering_pipeline(numeric_features, categorical_features)
        preprocessor.fit(df)
    
    # Aplicar transformaciones
    df_transformed = preprocessor.transform(df)
    
    # Obtener nombres de características
    feature_names = numeric_features.copy()
    
    if categorical_features and hasattr(preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
        cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names.extend(cat_features)
    
    # Crear DataFrame con nombres de columnas
    df_final = pd.DataFrame(df_transformed, columns=feature_names)
    
    # Agregar target si existe
    if target is not None:
        df_final['target'] = target
    
    return df_final, preprocessor

def main():
    """Función principal"""
    try:
        print("\nIniciando procesamiento de datos...")
        
        # Crear directorios necesarios
        create_directories()
        
        # Cargar datos de entrenamiento
        print("\nCargando datos de entrenamiento...")
        df_train = load_data('train_infarto.csv')
        
        # Procesar datos de entrenamiento
        print("\nProcesando datos de entrenamiento...")
        df_train_processed, preprocessor = process_dataset(df_train, train_mode=True)
        
        # Guardar datos procesados de entrenamiento
        train_output = os.path.join('data', 'processed', 'train_processed.csv')
        df_train_processed.to_csv(train_output, index=False)
        print(f"\nDatos de entrenamiento procesados guardados en: {train_output}")
        
        # Guardar preprocessor
        preprocessor_file = os.path.join('models', 'preprocessor.joblib')
        joblib.dump(preprocessor, preprocessor_file)
        print(f"\nPreprocessor guardado en: {preprocessor_file}")
        
        # Cargar y procesar datos de test si existen
        test_file = os.path.join('data', 'raw', 'test_infarto.csv')
        if os.path.exists(test_file):
            print("\nCargando datos de test...")
            df_test = load_data('test_infarto.csv')
            
            print("\nProcesando datos de test...")
            df_test_processed, _ = process_dataset(df_test, preprocessor=preprocessor, train_mode=False)
            
            # Guardar datos procesados de test
            test_output = os.path.join('data', 'processed', 'test_processed.csv')
            df_test_processed.to_csv(test_output, index=False)
            print(f"\nDatos de test procesados guardados en: {test_output}")
        
        print("\nProcesamiento completado exitosamente")
        return 0
        
    except Exception as e:
        print(f"\nError en el procesamiento: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
