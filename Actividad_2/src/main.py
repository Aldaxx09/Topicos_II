import sys
import os
import warnings
import logging


#Silenciar Advertencias (Warnings) 
warnings.filterwarnings("ignore")
# Solo mostrarán errores críticos, ocultando los mensajes INFO
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy").setLevel(logging.ERROR)


import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time
from typing import List, Dict, Any

# Importamos las clases modulares
from module_data import DataProcessor
from module_ml import ModelEvaluator

# =CONFIGURACIÓN DEL EXPERIMENTO BASE =
SCALING_STRATEGY = 'minmax' # 'standard' o 'minmax'
FEATURE_SELECTION_STRATEGY = 'rfecv' # 'none', 'pca', 'rfecv'

# Lista de modelos a evaluar
MODELS_TO_RUN: List[str] = [
    'LogisticRegression',
    'RandomForest',
    'GaussianNB',
    'MLP',
    "Bagging",
    "AdaBoost",
    "GradientBoosting",
    "DecisionTree"
]

def main():
    print("--- Iniciando Pipeline Modular con MLflow (Multi-Modelo) ---")
    
    # Configurar base de datos SQLite
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    except Exception:
        pass # Ignorar si ya está configurado

    # Configurar el nombre del experimento en MLflow
    EXPERIMENT_NAME = "WIDS_Experimentacion_Modular_MultiModelo"
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Lista para almacenar los resultados comparativos
    comparison_results: List[Dict[str, Any]] = []
    
    # 1. Instanciar Procesador de Datos
    processor = DataProcessor(seed=42)
    
    # 2. Obtener datos PROCESADOS y DIVIDIDOS
    print(f"\n>>> 1. Procesamiento y Split (Estrategia: {SCALING_STRATEGY} + {FEATURE_SELECTION_STRATEGY})...")
    try:
        X_train, X_test, y_train, y_test = processor.get_processed_data(
            scaler_method=SCALING_STRATEGY,
            feature_method=FEATURE_SELECTION_STRATEGY
        )
    except Exception as e:
        print(f"ERROR al cargar/procesar datos: {e}")
        return

    n_features = X_train.shape[1]
    print(f"Datos Listos. Features finales: {n_features}")

    # Crear 'input_example' para la firma del modelo en MLflow
    input_example = X_train.head(1)

    # 3. LOOP DE EXPERIMENTACIÓN Y EVALUACIÓN
    print("\n>>> 2. Iniciando Entrenamiento y Evaluación por Modelo...")
    
    for model_name in MODELS_TO_RUN:
        run_name = f"{model_name}_{FEATURE_SELECTION_STRATEGY}"
        print(f"\n===> Ejecutando Experimento: {run_name}")
        
        with mlflow.start_run(run_name=run_name):
            
            # --- A. REGISTRO DE PARÁMETROS ---
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("scaling_strategy", SCALING_STRATEGY)
            mlflow.log_param("feature_selection", FEATURE_SELECTION_STRATEGY)
            mlflow.log_param("n_features_final", n_features)

            # Instanciar y Evaluar el modelo
            try:
                model_instance = ModelEvaluator.get_model_instance(model_name)
                evaluator = ModelEvaluator(model_instance)
                
                # Medir el tiempo de entrenamiento
                start_time = time.time()
                evaluator.train(X_train, y_train)
                end_time = time.time()
                train_time = end_time - start_time
                
                # Obtener métricas
                metrics = evaluator.evaluate(X_test, y_test)
                
                # --- B. REGISTRO DE MÉTRICAS Y TIEMPO ---
                mlflow.log_metrics(metrics)
                mlflow.log_metric("training_time_sec", train_time)
                print(f" Métricas registradas en MLflow: {metrics}")
                print(f" Tiempo de entrenamiento: {train_time:.2f} segundos.")
                
                # REGISTRO DEL MODELO
                mlflow.sklearn.log_model(
                    sk_model=model_instance, 
                    name=model_name,
                    input_example=input_example
                )
                
                # Almacenar resultados para la tabla comparativa
                comparison_results.append({
                    "Modelo": model_name,
                    "Accuracy": metrics.get("accuracy", np.nan),
                    "ROC-AUC": metrics.get("roc_auc", np.nan),
                    "Tiempo (s)": f"{train_time:.2f}",
                })

            except ValueError as ve:
                print(f" SKIPPED: {model_name}. Error: {ve}")
                mlflow.log_param("status", "SKIPPED")
            except Exception as e:
                print(f"ERROR inesperado durante la ejecución de {model_name}: {e}")
                mlflow.log_param("status", "FAILED")
                
    # 4. GENERAR TABLA COMPARATIVA
    if comparison_results:
        print("\n" + "="*70)
        print("                 TABLA DE COMPARACIÓN DE EXPERIMENTOS")
        print("="*70)
        df_results = pd.DataFrame(comparison_results)
        df_results = df_results.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
        print(df_results.to_markdown(index=False, floatfmt=".4f"))
        print("="*70)
    else:
        print("\nNo se pudieron generar resultados de comparación.")

    print("\n--- Pipeline de Experimentación finalizado exitosamente. ---")

if __name__ == "__main__":
    main()
# mlflow ui