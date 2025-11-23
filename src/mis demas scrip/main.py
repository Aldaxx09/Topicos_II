# main.py

import sys
import module_data
import module_ml
# No necesitamos LogisticRegression aquí, se maneja en module_ml
# No necesitamos pprint ni numpy aquí

def main():
    """
    Orquesta el pipeline completo de ML:
    1. Carga datos
    2. Divide datos (Train/Test)
    3. Define features y crea preprocesador
    4. Define modelos y sus grillas
    5. Ejecuta un ciclo de experimentación (GridSearch + Evaluación)
    """
    print("--- INICIANDO PIPELINE DE ML (WIDS 2024) ---")
    
    # 1. Carga de Datos
    try:
        df = module_data.load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Asegúrate de que el archivo 'training.csv' esté en la carpeta 'data/'.", file=sys.stderr)
        return
    except Exception as e:
        print(f"Un error inesperado ocurrió cargando datos: {e}", file=sys.stderr)
        return

    # 2. División de Datos (Train/Test Split)
    try:
        X_train, X_test, y_train, y_test = module_data.get_train_test_split(df)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Asegúrate que la variable 'TARGET_COLUMN' en 'module_data.py' es correcta.", file=sys.stderr)
        return

    # 3. Definición de Features y Preprocesador
    num_feats, cat_feats = module_data.get_feature_definitions(X_train)
    
    if not num_feats and not cat_feats:
        print("Advertencia: No se encontraron features numéricas ni categóricas.", file=sys.stderr)
        print("Revisa 'get_feature_definitions()' en 'module_data.py'.", file=sys.stderr)
        # No detenemos la ejecución, puede que ColumnTransformer maneje listas vacías
        
    preprocessor = module_data.create_preprocessor(num_feats, cat_feats)
    
    # 4. Definición de Modelos y Grillas
    models, param_grids = module_ml.get_models_and_grids()
    
    best_models_trained = {} # Un diccionario para guardar los modelos entrenados
    
    # 5. Ciclo de Experimentación
    print("\n--- INICIANDO CICLO DE EXPERIMENTACIÓN ---")
    for name, model in models.items():
        print(f"\n=========================================")
        print(f"         Modelo: {name}                  ")
        print(f"=========================================")
        
        # 5.1. Construir el pipeline completo (Preprocesador + Modelo)
        pipeline = module_ml.build_model_pipeline(preprocessor, model)
        
        # 5.2. Obtener su grilla de parámetros
        param_grid = param_grids[name]
        
        # 5.3. Correr el experimento (GridSearch + Evaluación)
        best_model = module_ml.run_experiment(
            pipeline=pipeline,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        
        best_models_trained[name] = best_model

    print("\n--- PIPELINE COMPLETADO ---")
    print(f"Se entrenaron y evaluaron {len(best_models_trained)} modelos.")
    print(f"Modelos entrenados: {list(best_models_trained.keys())}")


if __name__ == "__main__":
    main()