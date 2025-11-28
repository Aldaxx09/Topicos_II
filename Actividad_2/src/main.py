
import os
import sys
import warnings
import time
from typing import List, Dict, Any

# --- Lineas para limpiar la consola ---
os.environ["TQDM_DISABLE"] = "1"  # Apaga la barra de 'Downloading artifacts'
warnings.filterwarnings("ignore") # Apaga los textos amarillos y advertencias

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from module_data import DataProcessor
from module_ml import ModelEvaluator

# ==CONFIGURACIÓN ===
SCALING_STRATEGY = 'minmax' # 'standard' o 'minmax'
FEATURE_SELECTION_STRATEGY = 'rfecv' # 'none', 'pca', 'rfecv'


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
# =================================================

def main():
    print("--- Iniciando Pipeline Modular con MLflow (Multi-Modelo) ---")
    
    # Esto guarda 'mlruns' una carpeta atrás (fuera de src)
    mlflow.set_tracking_uri("file:../mlruns")
    
    EXPERIMENT_NAME = "WIDS_Experimentacion_Modular_MultiModelo"
    mlflow.set_experiment(EXPERIMENT_NAME)

    comparison_results: List[Dict[str, Any]] = []
    
    # 1. Procesamiento
    processor = DataProcessor(seed=42)
    
    print(f"\n>>> 1. Procesamiento y Split (Estrategia: {SCALING_STRATEGY} + {FEATURE_SELECTION_STRATEGY})...")
    try:
        X_train, X_test, y_train, y_test = processor.get_processed_data(
            scaler_method=SCALING_STRATEGY,
            feature_method=FEATURE_SELECTION_STRATEGY
        )
    except Exception as e:
        print(f"ERROR al cargar datos: {e}")
        return

    n_features = X_train.shape[1]
    print(f"Datos Listos. Features finales: {n_features}")

    input_example = X_train.head(1)

    # 2. Entrenamiento
    print("\n>>> 2. Iniciando Entrenamiento y Evaluación por Modelo...")
    
    for model_name in MODELS_TO_RUN:
        run_name = f"{model_name}_{FEATURE_SELECTION_STRATEGY}"
        print(f"\n===> Ejecutando Experimento: {run_name}")
        
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("scaling_strategy", SCALING_STRATEGY)
            mlflow.log_param("n_features", n_features)

            try:
                model_instance = ModelEvaluator.get_model_instance(model_name)
                evaluator = ModelEvaluator(model_instance)
                
                # Entrenar
                start = time.time()
                evaluator.train(X_train, y_train)
                train_time = time.time() - start
                
                # Evaluar
                metrics = evaluator.evaluate(X_test, y_test)
                
                # Logs
                mlflow.log_metrics(metrics)
                mlflow.log_metric("training_time_sec", train_time)
                
                print(f"Métricas registradas: {metrics}")
                print(f"Tiempo: {train_time:.2f} s")
                
                # Guardar Modelo
                mlflow.sklearn.log_model(
                    sk_model=model_instance, 
                    name=model_name,
                    input_example=input_example
                )
                
                comparison_results.append({
                    "Modelo": model_name,
                    "Accuracy": metrics.get("accuracy", np.nan),
                    "ROC-AUC": metrics.get("roc_auc", np.nan),
                    "Tiempo (s)": f"{train_time:.2f}",
                })

            except Exception as e:
                print(f"ERROR en {model_name}: {e}")
                
    # 3. Tabla Final
    if comparison_results:
        print("\n" + "="*70)
        print("                 TABLA DE COMPARACIÓN DE EXPERIMENTOS")
        print("="*70)
        df = pd.DataFrame(comparison_results)
        df = df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
        print(df.to_markdown(index=False, floatfmt=".4f"))
        print("="*70)

    print("\n--- Pipeline finalizado exitosamente. ---")

if __name__ == "__main__":
    main()