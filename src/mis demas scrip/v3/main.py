
import pandas as pd
import sys
# Importamos nuestros módulos funcionales
from module_data import Dataset
from module_ml import Model

# ==========================================================
#              CONFIGURACIÓN DEL EXPERIMENTO (4 Decisiones)
# ==========================================================

# 1. Elegir método de limpieza: 
# Opciones: 'dropna', 'mean'/'most_frequent', 'knn'
CLEANING_STRATEGY = 'mean' 

# 2. Elegir método de escalamiento: 
# Opciones: 'standard' (StandardScaler), 'minmax' (MinMaxScaler), 'none'
SCALING_STRATEGY = 'standard' 

# 3. Elegir método de selección de características: 
# Opciones: 'rfecv', 'pca', 'none'
FEATURE_SELECTION_STRATEGY = 'rfecv'

# 4. Elegir el modelo a usar: 
# Opciones: 'LogisticRegression', 'RandomForest', 'SVC', 'DecisionTree', 'MLP', 'KNN'
MODEL_NAME = 'LogisticRegression' 

# ==========================================================

def main():
    """
    Función principal que orquesta el pipeline de ML.
    """
    print("--- Iniciando Pipeline Modular de Machine Learning ---")
    
    # 1. CARGA Y PREPROCESAMIENTO (Estrategias 1 y 2)
    data_handler = Dataset(num_samples=None, seed=42)
    
    X_processed, y_target = data_handler.preprocess_data(
        clean_method=CLEANING_STRATEGY, 
        scaler_method=SCALING_STRATEGY
    )
    
    # 2. SELECCIÓN DE CARACTERÍSTICAS (Estrategia 3)
    X_final = data_handler.apply_feature_selection(
        X=X_processed, 
        y=y_target, 
        feature_method=FEATURE_SELECTION_STRATEGY
    )
    
    print(f"Dimensiones finales para el modelado (X): {X_final.shape}, (y): {y_target.shape}")
    
    # 3. MODELADO Y EVALUACIÓN (Estrategia 4: Modelo)
    
    model_instance = Model.get_model_instance(MODEL_NAME)
    
    ml_experiment = Model(X=X_final, y=y_target, seed=42)
    
    # Llama a train_and_evaluate SIN el parámetro SMOTE
    ml_experiment.train_and_evaluate(
        model=model_instance
    )
    
    print("\n--- Pipeline finalizado exitosamente. ---")

if __name__ == "__main__":
    main()