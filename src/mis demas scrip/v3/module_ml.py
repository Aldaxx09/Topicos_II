
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
# Modelos de clasificación configurables
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier 
# Se eliminan las importaciones de imblearn.

class Model:
    def __init__(self, X: pd.DataFrame, y: pd.Series, seed: int = 42):
        """Inicializa la clase Model con datos y semilla."""
        self.X = X
        self.y = y
        self.seed = seed

    def split(self, train_size: float = 0.8):
        """Divide los datos en conjuntos de entrenamiento y prueba, estratificando por 'y' [18]."""
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            train_size=train_size,
                                                            random_state=self.seed,
                                                            stratify=self.y # Mantener la proporción de clases
                                                            )
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, model):
        """
        Entrena y evalúa el modelo, 
        :param model: Instancia del modelo a entrenar (Decisión final).
        """
        X_train, X_test, y_train, y_test = self.split() # División de datos

        print(f"Iniciando entrenamiento del modelo {model.__class__.__name__}...")
        
        model.fit(X_train, y_train) 
        print("Entrenamiento completado.")

        # --- Evaluación ---
        y_pred = model.predict(X_test) 
        
        print("\n--- Métricas Relevantes en el Conjunto de Prueba ---")
        accuracy = accuracy_score(y_test, y_pred) # Exactitud
        roc_score = roc_auc_score(y_test, y_pred) # ROC AUC 
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC_AUC Score: {roc_score:.4f}")
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred))

    @staticmethod
    def get_model_instance(model_name: str):
        """Factoría estática para obtener una instancia del modelo basado en el nombre."""
        if model_name == 'LogisticRegression':
            return LogisticRegression(random_state=42, max_iter=10000, solver='liblinear')
        elif model_name == 'RandomForest':
            return RandomForestClassifier(random_state=42, n_estimators=500)
        elif model_name == 'SVC':
            return SVC(random_state=42, kernel='linear', probability=True)
        elif model_name == 'DecisionTree':
            return DecisionTreeClassifier(random_state=42)
        elif model_name == 'MLP':
            # Red Neuronal Multicapa
            return MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)
        elif model_name == 'KNN':
            return KNeighborsClassifier()
        else:
            # Opción para añadir más modelos 
            raise ValueError(f"Modelo '{model_name}' no soportado o mal escrito.")
