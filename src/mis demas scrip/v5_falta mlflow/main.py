# main.py
import sys
import pandas as pd
import numpy as np
# Importamos las clases actualizadas
from module_data import DataProcessor
from module_ml import ModelEvaluator

# ================= CONFIGURACIÓN DEL EXPERIMENTO =================
CLEANING_STRATEGY = 'mean' 
SCALING_STRATEGY = 'standard' 
FEATURE_SELECTION_STRATEGY = 'rfecv' 
MODEL_NAME = 'LogisticRegression' 
# =================================================================

def main():
    print("--- Iniciando Pipeline Modular (Completo) ---")
    
    # 1. Instanciar Procesador de Datos
    processor = DataProcessor(seed=42)
    
    # 2. Obtener datos PROCESADOS y DIVIDIDOS (Validación Interna)
    print("\n>>> 1. Procesamiento y Split (Validación Interna)...")
    X_train, X_test, y_train, y_test = processor.get_processed_data(
        clean_method=CLEANING_STRATEGY,
        scaler_method=SCALING_STRATEGY,
        feature_method=FEATURE_SELECTION_STRATEGY
    )
    
    print(f"\nDatos Listos.")
    print(f"Dimensiones Train: {X_train.shape}, Dimensiones Test (Interno): {X_test.shape}")
    
    # 3. Entrenar y Evaluar (Validación Interna)
    print("\n>>> 2. Entrenamiento y Evaluación Interna...")
    model_instance = ModelEvaluator.get_model_instance(MODEL_NAME)
    evaluator = ModelEvaluator(model_instance)
    
    evaluator.train(X_train, y_train)
    evaluator.evaluate(X_test, y_test)
    
    # =============================================================
    # 4. PRÁCTICA DE INFERENCIA EN DATOS EXTERNOS ('test.csv')
    # =============================================================
    print("\n" + "="*50)
    print(">>> 3. Aplicando Modelo a Datos de Prueba Externos ('test.csv')")
    print("="*50)
    
    try:
        # A. Procesar el archivo test.csv
        X_external_test, patient_ids = processor.process_external_test_data()
        
        # B. Generar predicciones (Probabilidad)
        if hasattr(evaluator.model, "predict_proba"):
             predictions = evaluator.model.predict_proba(X_external_test)[:, 1]
        else:
             predictions = evaluator.model.predict(X_external_test)
            
        # C. Generar Reporte de Práctica
        results_df = pd.DataFrame({
            'patient_id': patient_ids,
            'prediction_probability': predictions
        })
        
        print(f"✅ Inferencia externa completada: {results_df.shape[0]} predicciones generadas.")
        print("El flujo completo (Carga -> FE -> Train -> Validación -> Inferencia) ha sido modularizado con éxito.")
        print("\nPrimeras 5 predicciones en test.csv:")
        print(results_df.head())
        
    except Exception as e:
        print(f"⚠️ ERROR al procesar 'test.csv'. Revise las rutas o si existe el archivo: {e}")
        
    print("\n--- Flujo finalizado. Arquitectura modular y robusta. ---")

if __name__ == "__main__":
    main()