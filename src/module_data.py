
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV # Para selección de características
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA # Para reducción de dimensionalidad

# Importar módulos propios
from module_path import train_data_path, test_data_path 

COL_DIAGPERIODL90D = "DiagPeriodL90D" # Variable objetivo

class Dataset:
    def __init__(self, num_samples: int = None, seed: int = 42):
        """
        Inicializa la clase Dataset.
        :param num_samples: Número de muestras a usar (para submuestreo).
        :param seed: Semilla para reproducibilidad.
        """
        self.num_samples = num_samples
        self.seed = seed

    def load_data(self):
        """Carga los datos brutos y elimina columnas irrelevantes."""
        # Lógica de carga y eliminación de columnas idéntica a la avanzada en clase
        train_path = train_data_path()
        df_train = pd.read_csv(train_path)
        
        # Columnas a eliminar (basado en el análisis exploratorio)
        cols_drop = ['patient_id', 'breast_cancer_diagnosis_desc', 
                     'metastatic_first_novel_treatment', 
                     'metastatic_first_novel_treatment_type']
        
        df_train = df_train.drop(columns=cols_drop)
        
        if self.num_samples is not None:
            df_train = df_train.sample(n=self.num_samples, random_state=self.seed)
            
        return df_train.reset_index(drop=True)

    def preprocess_data(self, clean_method: str = 'dropna', scaler_method: str = 'standard'):
        """
        Aplica limpieza, codificación (One-Hot Encoding) y escalamiento.
        :param clean_method: Estrategia para manejar nulos ('dropna', 'mean', 'most_frequent', 'knn').
        :param scaler_method: Estrategia de escalamiento ('standard', 'minmax', 'none').
        """
        df_train = self.load_data()
        
        # --- 1. FEATURE ENGINEERING (Ejemplo BMI Categorical) ---
        # Se recrea la columna categórica de BMI antes de la limpieza
        df_train['bmi_category'] = pd.cut(df_train['bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf], 
                                         labels=['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Extreme']).astype('object')
        df_train['payer_type'] = df_train['payer_type'].fillna('Unknown') # Imputación para 'payer_type'
        
        # --- 2. MANEJO DE VALORES NULOS (Basado en clean_method) ---
        print(f"Aplicando estrategia de limpieza: {clean_method}")
        if clean_method == 'dropna':
            # Elimina filas con nulos (Estrategia 1: Dropna)
            df_train = df_train.dropna()
        
        # Separar datos numéricos y categóricos para imputación
        numeric_cols = df_train.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_train.select_dtypes(include='object').columns.tolist()

        if clean_method in ['mean', 'most_frequent']:
            # Estrategia 2: Imputación Simple (media para num, moda para cat)
            num_imputer = SimpleImputer(strategy='mean')
            cat_imputer = SimpleImputer(strategy='most_frequent')
            
            df_train[numeric_cols] = num_imputer.fit_transform(df_train[numeric_cols])
            df_train[categorical_cols] = cat_imputer.fit_transform(df_train[categorical_cols])
        
        # --- 3. CODIFICACIÓN (One-Hot Encoding) ---
        # Se aplica OHE a las variables categóricas. Esto es CRÍTICO para ML.
        df_encoded = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True, dtype=int)
        
        if clean_method == 'knn':
            # Estrategia 3: KNN Imputer (Funciona DESPUÉS de OHE, solo en columnas numéricas) 
            knn_imputer = KNNImputer(n_neighbors=5)
            # Aplicamos KNN Imputer
            df_encoded = pd.DataFrame(knn_imputer.fit_transform(df_encoded), 
                                     columns=df_encoded.columns)
            # Nota: Esto podría generar problemas de rendimiento en datasets grandes, 
            # pero asegura la limpieza.
        
        # Asegurarse de que el target sea numérico (0/1) después del preprocesamiento
        y = df_encoded[COL_DIAGPERIODL90D]
        X = df_encoded.drop(columns=[COL_DIAGPERIODL90D])

        # --- 4. ESCALAMIENTO (Basado en scaler_method) ---
        print(f"Aplicando estrategia de escalamiento: {scaler_method}")
        
        if scaler_method == 'standard':
            scaler = StandardScaler() # Estandarización: media 0, desviación 1.
        elif scaler_method == 'minmax':
            scaler = MinMaxScaler() # Normalización: entre 0 y 1.
        else: # 'none'
            return X, y

        # Ajustar y transformar X
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled_df, y

    def apply_feature_selection(self, X: pd.DataFrame, y: pd.Series, feature_method: str = 'none'):
        """
        Aplica RFECV o PCA para reducir la dimensionalidad o seleccionar características.
        :param feature_method: Estrategia de selección ('rfecv', 'pca', 'none').
        """
        print(f"Aplicando selección de características: {feature_method}")

        if feature_method == 'rfecv':
            # Recursive Feature Elimination with Cross-Validation (RFECV)
            # Utiliza un modelo base (LR) para iterativamente eliminar las peores características.
            estimator = LogisticRegression(random_state=self.seed, max_iter=6000, solver='liblinear')
            rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='roc_auc', n_jobs=-1)
            rfecv.fit(X, y)
            X_reduced = X.loc[:, rfecv.support_]
            print(f"RFECV seleccionó {rfecv.n_features_} características óptimas.")
            return X_reduced
        
        elif feature_method == 'pca':
            # ... código de PCA (que asumimos que funciona correctamente)
            pca = PCA(n_components=0.95, random_state=self.seed)
            X_pca = pca.fit_transform(X)
            X_reduced = pd.DataFrame(X_pca, index=X.index)
            
            # LÍNEA CORREGIDA: Usar el índice [1] para reportar el número de características (columnas)
            print(f"PCA redujo las dimensiones de {X.shape[1]} a {X_reduced.shape[1]}.") 
            
            return X_reduced


'''
 Flujo Secuencial: El método preprocess_data encapsula la carga, la limpieza y el escalamiento. 
 Es crucial que la codificación (OHE) ocurra antes de que separe X e y y se aplique el escalamiento, 
 dado que el escalamiento (StandardScaler o MinMaxScaler) solo opera sobre características numéricas (lo que OHE genera).
• Decisión de Limpieza (clean_method): La elección de knn (KNNImputer) es importante, ya que solo puede aplicarse
a datos numéricos; por lo tanto, debe ejecutarse después de la codificación OHE. En contraste, dropna y la imputación simple
(mean/most_frequent) pueden aplicarse antes o después de OHE (pero aquí se aplican antes de OHE a los datos categóricos y 
numéricos por separado, como en su ejemplo).
• Selección de Características: El método apply_feature_selection recibe los datos (X, y) ya limpios y escalados. 
Esto es un paso de optimización que busca las variables más importantes (rfecv) o una representación comprimida (pca).
'''