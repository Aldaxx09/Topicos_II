
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
# Modelos de clasificación configurables
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier # MLP para el 'Neural Net'
# Manejo de Desbalance (requiere la librería imblearn)
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline as ImbPipeline

class Model:
    def __init__(self, X: pd.DataFrame, y: pd.Series, seed: int = 42):
        """Inicializa la clase Model con datos y semilla."""
        self.X = X
        self.y = y
        self.seed = seed

    def split(self, train_size: float = 0.8):
        """Divide los datos en conjuntos de entrenamiento y prueba, estratificando por 'y'."""
        # Se usa 'stratify=self.y' para asegurar que la proporción de clases se mantenga en train y test.
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            train_size=train_size,
                                                            random_state=self.seed,
                                                            stratify=self.y # Importante por el desbalance
                                                            )
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, model, use_smote: bool = False):
        """
        Entrena y evalúa el modelo, opcionalmente aplicando SMOTE en el conjunto de entrenamiento.
        :param model: Instancia del modelo a entrenar.
        :param use_smote: Si True, aplica SMOTE al X_train.
        """
        X_train, X_test, y_train, y_test = self.split() # División de datos

        if use_smote:
            print("Aplicando SMOTE para manejar el desbalance de clases...")
            sm = SMOTE(random_state=self.seed)
            # SMOTE solo se aplica a los datos de ENTRENAMIENTO 
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            print(f"Tamaño de entrenamiento antes de SMOTE: {len(y_train)}. Después: {len(y_train_res)}")
            
            # Usar los datos remuestreados
            X_train = X_train_res
            y_train = y_train_res

        print(f"Iniciando entrenamiento del modelo {model.__class__.__name__}...")
        
        # El entrenamiento se realiza sobre los datos (posiblemente remuestreados)
        model.fit(X_train, y_train) 
        print("Entrenamiento completado.")

        # --- Evaluación ---
        y_pred = model.predict(X_test) # Predicción sobre el conjunto de prueba (no visto)
        
        print("\n--- Métricas Relevantes en el Conjunto de Prueba ---")
        accuracy = accuracy_score(y_test, y_pred) # Exactitud [16]
        roc_score = roc_auc_score(y_test, y_pred) # ROC AUC (para clasificación binaria) [16]
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC_AUC Score: {roc_score:.4f}")
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred))

    @staticmethod
    def get_model_instance(model_name: str):
        """Factoría estática para obtener una instancia del modelo basado en el nombre."""
        if model_name == 'LogisticRegression':
            # Modelo de regresión logística, afinado para convergencia rápida
            return LogisticRegression(random_state=42, max_iter=10_000, solver='liblinear')
        elif model_name == 'RandomForest':
            # Modelo de ensamble (Bosque Aleatorio) [17]
            return RandomForestClassifier(random_state=42, n_estimators=500)
        elif model_name == 'SVC':
            # Máquinas de Vectores de Soporte (con kernel lineal, como se usa a menudo en texto) [18]
            return SVC(random_state=42, kernel='linear', probability=True)
        elif model_name == 'DecisionTree':
            # Árbol de Decisión individual [19]
            return DecisionTreeClassifier(random_state=42)
        elif model_name == 'MLP':
            # Perceptrón Multicapa (Red Neuronal Básica) [20]
            return MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)
        else:
            raise ValueError(f"Modelo '{model_name}' no soportado o mal escrito.")
        
        
'''
SMOTE (Synthetic Minority Oversampling Technique): SMOTE es una técnica para balancear clases en problemas de clasificación
cuando el dataset está desbalanceado. El módulo implementa la lógica para aplicarlo solo al conjunto de entrenamiento
(X_train, y_train), ya que si se aplicara al conjunto de prueba, se introduciría un sesgo irreal en la evaluación.
• Fábrica de Modelos (get_model_instance): Esta función estática permite que main.py solicite un modelo por su nombre
(LogisticRegression, RandomForest, etc.) sin tener que codificar la importación y la inicialización de la clase en main.py.

'''