# main.py
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

# Importamos las clases actualizadas
from module_data import DataProcessor
from module_ml import ModelEvaluator

# ================= CONFIGURACIÓN DEL EXPERIMENTO =================
# Puedes cambiar estos valores manualmente y volver a ejecutar para registrar nuevos experimentos
SCALING_STRATEGY = 'minmax' 
FEATURE_SELECTION_STRATEGY = 'rfecv' 
MODEL_NAME = 'GaussianNB' 
# =================================================================

def main():
    print("--- Iniciando Pipeline Modular con MLflow ---")
    
    # Configurar el nombre del experimento en MLflow
    mlflow.set_experiment("WIDS_Experimentacion_Modular")
    
    # Iniciar el tracking del experimento
    with mlflow.start_run(run_name=f"{MODEL_NAME}_{FEATURE_SELECTION_STRATEGY}"):
        
        # -------------------------------------------------
        # A. REGISTRO DE PARÁMETROS (Configuración)
        # -------------------------------------------------
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("scaling_strategy", SCALING_STRATEGY)
        mlflow.log_param("feature_selection", FEATURE_SELECTION_STRATEGY)
        
        # 1. Instanciar Procesador de Datos
        processor = DataProcessor(seed=42)
        
        # 2. Obtener datos PROCESADOS y DIVIDIDOS (Validación Interna)
        print("\n>>> 1. Procesamiento y Split (Validación Interna)...")
        X_train, X_test, y_train, y_test = processor.get_processed_data(
            scaler_method=SCALING_STRATEGY,
            feature_method=FEATURE_SELECTION_STRATEGY
        )
        
        # Registrar cuántas features quedaron
        mlflow.log_param("n_features_final", X_train.shape[1])
        print(f"\nDatos Listos. Features finales: {X_train.shape[1]}")
        
        # 3. Entrenar y Evaluar
        print("\n>>> 2. Entrenamiento y Evaluación Interna...")
        model_instance = ModelEvaluator.get_model_instance(MODEL_NAME)
        evaluator = ModelEvaluator(model_instance)
        
        evaluator.train(X_train, y_train)
        
        # Obtenemos las métricas del evaluador (gracias al cambio en module_ml.py)
        metrics = evaluator.evaluate(X_test, y_test)
        
        # -------------------------------------------------
        # B. REGISTRO DE MÉTRICAS (Resultados)
        # -------------------------------------------------
        mlflow.log_metrics(metrics)
        print(f"✅ Métricas registradas en MLflow: {metrics}")
        
        # -------------------------------------------------
        # C. REGISTRO DEL MODELO (Artefacto)
        # -------------------------------------------------
        mlflow.sklearn.log_model(evaluator.model, "model")
        print("✅ Modelo guardado en MLflow.")

    print("\n--- Experimento finalizado exitosamente. ---")

if __name__ == "__main__":
    main()

# mlflow ui