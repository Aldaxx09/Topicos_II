# module_ml.py

import pandas as pd
from typing import Dict, Any, Tuple  # <--- CORRECCIÓN: Imports de typing

# scikit-learn: modelos y métricas
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.compose import ColumnTransformer # <--- CORRECCIÓN: Importamos ColumnTransformer

# Modelos (¡Los 4 de tu notebook!)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Constantes
RANDOM_SEED = 42

def build_model_pipeline(
    preprocessor: ColumnTransformer, # <--- CORRECCIÓN: Anotación de tipo
    model
) -> Pipeline:
    """
    Crea un pipeline de Scikit-learn uniendo el preprocesador y el modelo.
    """
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model) # 'classifier' es el nombre clave para el param_grid
    ])

def get_models_and_grids() -> Tuple[Dict, Dict]: # <--- CORRECCIÓN: Anotación de tipo
    """
    Define los 4 modelos y sus grillas de parámetros (param_grids)
    basado en Wids2024.ipynb.
    """
    print("Definiendo modelos y grillas de parámetros...")
    
    # 1. Definición de Modelos
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_SEED),
        'MLP (Neural Net)': MLPClassifier(random_state=RANDOM_SEED, max_iter=500, early_stopping=True)
    }
    
    # 2. Definición de Grillas de Parámetros (basadas en tu notebook)
    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.1, 1.0, 10],
            'classifier__solver': ['liblinear', 'saga']
        },
        'Random Forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10]
        },
        'Decision Tree': {
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_leaf': [1, 5]
        },
        'MLP (Neural Net)': {
            'classifier__hidden_layer_sizes': [(50,), (100,)],
            'classifier__alpha': [0.001, 0.01]
        }
    }
    
    return models, param_grids

def run_experiment(
    pipeline: Pipeline, 
    param_grid: Dict[str, Any], 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    scoring: str = 'roc_auc'
):
    """
    Ejecuta el GridSearchCV para un modelo y evalúa el mejor estimador.
    """
    print(f"Ejecutando GridSearchCV (optimizando para '{scoring}')...")
    
    # 1. Configurar y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, # 5-fold cross-validation
        scoring=scoring, 
        n_jobs=-1, # Usar todos los cores
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # 2. Obtener el mejor modelo
    best_model = grid_search.best_estimator_
    
    print("\n--- Resultados del Experimento ---")
    print(f"Mejores parámetros encontrados:")
    print(grid_search.best_params_)
    print(f"Mejor score CV ({scoring}): {grid_search.best_score_:.4f}")
    
    # 3. Evaluar en el conjunto de Test
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] # Probabilidad de la clase 1

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\n--- Métricas en Conjunto de Test ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print("\nReporte de Clasificación (Test):")
    print(classification_report(y_test, y_pred))
    
    return best_model