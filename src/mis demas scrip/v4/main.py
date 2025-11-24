# main.py
import sys
import pandas as pd
# Importamos las clases actualizadas
from module_data import DataProcessor
from module_ml import ModelEvaluator

# ================= CONFIGURACIÓN =================
CLEANING_STRATEGY = 'mean' 
SCALING_STRATEGY = 'standard' 
FEATURE_SELECTION_STRATEGY = 'rfecv' 
MODEL_NAME = 'LogisticRegression' 
# =================================================

def main():
    print("--- Iniciando Pipeline Avanzado (WIDS 2024 Refactorizado) ---")
    
    # 1. Instanciar Procesador de Datos
    # Este objeto contiene toda tu nueva lógica de edad, contaminación, zip, etc.
    processor = DataProcessor(seed=42)
    
    # 2. Obtener datos PROCESADOS y DIVIDIDOS
    # DataProcessor se encarga internamente de no mezclar train y test
    print(">>> Ejecutando Ingeniería de Características y Preprocesamiento...")
    X_train, X_test, y_train, y_test = processor.get_processed_data(
        clean_method=CLEANING_STRATEGY,
        scaler_method=SCALING_STRATEGY,
        feature_method=FEATURE_SELECTION_STRATEGY
    )
    
    print(f"\nDatos Listos.")
    print(f"Dimensiones Train: {X_train.shape}")
    print(f"Dimensiones Test:  {X_test.shape}")
    
    # Verificar si tenemos columnas nuevas
    print(f"Ejemplo de columnas: {list(X_train.columns["zip_region_freq"])}")
    
    # 3. Instanciar Modelo
    model_instance = ModelEvaluator.get_model_instance(MODEL_NAME)
    
    # 4. Entrenar y Evaluar
    evaluator = ModelEvaluator(model_instance)
    
    evaluator.train(X_train, y_train)
    evaluator.evaluate(X_test, y_test)
    
    print("\n--- Pipeline finalizado exitosamente. ---")

if __name__ == "__main__":
    main()