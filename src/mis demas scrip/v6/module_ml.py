# module_ml.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class ModelEvaluator:
    def __init__(self, model):
        """
        Recibe una instancia de modelo ya configurada.
        """
        self.model = model

    def train(self, X_train, y_train):
        """Entrena el modelo usando SOLO el set de entrenamiento."""
        print(f"Entrenando {self.model.__class__.__name__}...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Predice y evalúa usando el set de prueba."""
        print("Evaluando modelo...")
        y_pred = self.model.predict(X_test)
        
        # Calcular ROC-AUC de forma robusta
        if hasattr(self.model, "predict_proba"):
            try:
                # La mayoría de modelos
                y_prob = self.model.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, y_prob)
            except:
                # Algunos modelos pueden fallar si solo hay 1 clase en el batch
                roc = 0.5
        else:
            # Modelos que no tienen predict_proba (como SVM sin flag)
            try:
                # Intentamos usar decision_function si existe (ej. SVM)
                y_prob = self.model.decision_function(X_test)
                roc = roc_auc_score(y_test, y_prob)
            except:
                roc = 0.5
            
        acc = accuracy_score(y_test, y_pred)
        
        print("\n--- Resultados de Evaluación ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC:  {roc:.4f}")
        print("\n" + classification_report(y_test, y_pred))

    @staticmethod
    def get_model_instance(model_name: str, seed: int = 42):
        """
        Factory Method extendido con todos los modelos comunes en notebooks de Ciencia de Datos.
        """
        if model_name == 'LogisticRegression':
            return LogisticRegression(
                random_state=seed, 
                max_iter=10000, 
                solver='liblinear', 
                class_weight='balanced' # Crucial para tu desbalance
            )
            
        elif model_name == 'RandomForest':
            return RandomForestClassifier(
                random_state=seed, 
                n_estimators=200, 
                class_weight='balanced'
            )
        
        elif model_name == 'GradientBoosting':
            return GradientBoostingClassifier(
                random_state=seed,
                n_estimators=200,
                learning_rate=0.1
            )
            
        elif model_name == 'AdaBoost':
            return AdaBoostClassifier(
                random_state=seed,
                n_estimators=100
            )
            
        elif model_name == 'Bagging':
            return BaggingClassifier(
                random_state=seed,
                n_estimators=50
            )

        elif model_name == 'SVC':
            return SVC(
                random_state=seed, 
                probability=True, # Necesario para curvas ROC
                class_weight='balanced'
            )

        elif model_name == 'DecisionTree':
            return DecisionTreeClassifier(
                random_state=seed, 
                class_weight='balanced'
            )

        elif model_name == 'MLP':
            return MLPClassifier(
                random_state=seed, 
                max_iter=1000,
                hidden_layer_sizes=(100, 50), # Arquitectura típica básica
                early_stopping=True
            )

        elif model_name == 'KNN':
            return KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
            
        elif model_name == 'GaussianNB':
            return GaussianNB()

        else:
            raise ValueError(f"Modelo '{model_name}' no soportado.")
        