
import pandas as pd
import sys
# Importamos nuestros módulos 
from module_data import Dataset
from module_ml import Model



# 1. Elegir método de limpieza: 
#    Opciones: 'dropna', 'mean'/'most_frequent', 'knn'
CLEANING_STRATEGY = 'knn' 

# 2. Elegir método de escalamiento: 
#    Opciones: 'standard' (StandardScaler), 'minmax' (MinMaxScaler), 'none'
SCALING_STRATEGY = 'standard' 

# 3. Elegir método de selección de características: 
#    Opciones: 'rfecv', 'pca', 'none'
FEATURE_SELECTION_STRATEGY = 'rfecv'

# 4. Elegir si usar SMOTE: 
#    Opciones: True, False
USE_SMOTE = True 

# 5. Elegir el modelo a usar: 
#    Opciones: 'LogisticRegression', 'RandomForest', 'SVC', 'DecisionTree', 'MLP'
MODEL_NAME = 'LogisticRegression' 

# ==========================================================

def main():
    """
    Función principal que orquesta el pipeline de ML.
    Sigue el flujo: Carga -> Preprocesamiento -> Selección de Características -> 
    Modelado (Split, SMOTE, Entrenamiento, Evaluación).
    """
    print("--- Iniciando Pipeline Modular de Machine Learning ---")
    
    # 1. CARGA Y PREPROCESAMIENTO
    data_handler = Dataset(num_samples=None, seed=42)
    
    # Se obtienen X y Y preprocesados y escalados
    X_processed, y_target = data_handler.preprocess_data(
        clean_method=CLEANING_STRATEGY, 
        scaler_method=SCALING_STRATEGY
    )
    
    # 2. SELECCIÓN DE CARACTERÍSTICAS
    X_final = data_handler.apply_feature_selection(
        X=X_processed, 
        y=y_target, 
        feature_method=FEATURE_SELECTION_STRATEGY
    )
    
    print(f"Dimensiones finales para el modelado (X): {X_final.shape}, (y): {y_target.shape}")
    
    
    
    # 3. MODELADO Y EVALUACION
    
    # Obtener el modelo deseado
    model_instance = Model.get_model_instance(MODEL_NAME)

    # Inicializar el manejador de modelos con los datos finales
    ml_experiment = Model(X=X_final, y=y_target, seed=42)
    
    # Entrenar y evaluar con las estrategias configuradas
    ml_experiment.train_and_evaluate(
        model=model_instance, 
        use_smote=USE_SMOTE
    )
    
    print("\n--- Pipeline finalizado exitosamente. ---")

if __name__ == "__main__":
    main()