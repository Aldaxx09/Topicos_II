# module_data.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from joblib import dump, load
import os

# Importar módulos propios
from module_path import train_data_path, test_data_path 

COL_DIAGPERIODL90D = "DiagPeriodL90D" # Variable objetivo
RFECV_MODEL_PATH = "rfecv_model.joblib" # Ruta para guardar/cargar RFECV

class Dataset:
    def __init__(self, num_samples: int = None, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed

    def load_data(self):
        """Carga los datos brutos y elimina columnas irrelevantes."""
        train_path = train_data_path()
        df_train = pd.read_csv(train_path)
        
        # Columnas a eliminar (basado en Winds2024.ipynb)
        cols_drop = ['patient_id', 'breast_cancer_diagnosis_desc', 
                     'metastatic_first_novel_treatment', 
                     'metastatic_first_novel_treatment_type']
        
        df_train = df_train.drop(columns=cols_drop, errors='ignore')
        
        if self.num_samples is not None:
            df_train = df_train.sample(n=self.num_samples, random_state=self.seed)
            
        return df_train.reset_index(drop=True)

    def preprocess_data(self, clean_method: str = 'mean', scaler_method: str = 'standard'):
        """Aplica Ingeniería de Características, limpieza (Decisión 1) y escalamiento (Decisión 2)."""
        df_train = self.load_data()
        
        # --- 1. INGENIERÍA DE CARACTERÍSTICAS (Refleja Winds2024 optimizado) ---
        
        # 1.1 Manejo de BMI: Creación de variable categórica
        bins_bmi = [0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf]
        labels_bmi = ['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Extreme']
        df_train['bmi_category'] = pd.cut(df_train['bmi'], bins=bins_bmi, labels=labels_bmi, right=False).astype('object')
        df_train['bmi_category'] = df_train['bmi_category'].fillna('Unknown')
        df_train.drop(columns=['bmi'], errors='ignore', inplace=True)
        
        # 1.2 Imputación de 'payer_type' y 'patient_race' antes de OHE
        df_train['payer_type'] = df_train['payer_type'].fillna('Unknown')
        df_train['patient_race'] = df_train['patient_race'].fillna('Unknown')
        
        # 1.3 Binarización de Edad (Age Binning), eliminando la original
        df_train['age_18_29'] = np.where((df_train['patient_age']>=18) & (df_train['patient_age']<30), 1, 0)
        df_train['age_30_44'] = np.where((df_train['patient_age']>=30) & (df_train['patient_age']<45), 1, 0)
        df_train['age_45_59'] = np.where((df_train['patient_age']>=45) & (df_train['patient_age']<60), 1, 0)
        df_train['age_60_74'] = np.where((df_train['patient_age']>=60) & (df_train['patient_age']<75), 1, 0)
        df_train['age_75_plus'] = np.where(df_train['patient_age']>=75, 1, 0)
        df_train.drop(columns=['patient_age'], errors='ignore', inplace=True)
        
        # 1.4 Frequency Encoding de Región ZIP (zip_region_freq)
        df_train['zip_region'] = df_train['patient_zip3'].astype(str).str[:3]
        frecuencias = df_train['zip_region'].value_counts()
        df_train['zip_region_freq'] = df_train['zip_region'].map(frecuencias)
        
        # 1.5 Eliminación de variables auxiliares y redundantes
        df_train.drop(columns=['patient_gender', 'zip_region', 'patient_zip3'], errors='ignore', inplace=True)
        
        # --- 2. MANEJO DE VALORES NULOS RESTANTES (Decisión 1) y CODIFICACIÓN ---
        
        print(f"Aplicando estrategia de limpieza: {clean_method}")
        
        if clean_method == 'dropna':
            df_train = df_train.dropna().reset_index(drop=True)
        
        numeric_cols = df_train.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_train.select_dtypes(include='object').columns.tolist()

        if clean_method in ['mean', 'most_frequent']:
            # Imputación Simple
            num_imputer = SimpleImputer(strategy='mean')
            cat_imputer = SimpleImputer(strategy='most_frequent')
            
            num_cols_with_nulls = [col for col in numeric_cols if df_train[col].isnull().any()]
            cat_cols_with_nulls = [col for col in categorical_cols if df_train[col].isnull().any()]
            
            if num_cols_with_nulls:
                df_train[num_cols_with_nulls] = num_imputer.fit_transform(df_train[num_cols_with_nulls])
            if cat_cols_with_nulls:
                df_train[cat_cols_with_nulls] = cat_imputer.fit_transform(df_train[cat_cols_with_nulls])

        # --- 3. CODIFICACIÓN (One-Hot Encoding) ---
        df_encoded = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True, dtype=int)
        
        if clean_method == 'knn':
            # KNN Imputer (Aplicado después de OHE, sobre datos numéricos)
            print("Aplicando imputación KNN...")
            knn_imputer = KNNImputer(n_neighbors=5)
            
            # Separar temporalmente la columna target
            y_temp = df_encoded[COL_DIAGPERIODL90D]
            X_data = df_encoded.drop(columns=[COL_DIAGPERIODL90D], errors='ignore')
            
            X_imputado_np = knn_imputer.fit_transform(X_data)
            df_encoded = pd.DataFrame(X_imputado_np, columns=X_data.columns, index=X_data.index)
            # Reinsertar la columna target
            df_encoded[COL_DIAGPERIODL90D] = y_temp
        
        # Separar X e Y
        y = df_encoded[COL_DIAGPERIODL90D]
        X = df_encoded.drop(columns=[COL_DIAGPERIODL90D])

        # --- 4. ESCALAMIENTO (Decisión 2) ---
        print(f"Aplicando estrategia de escalamiento: {scaler_method}")
        
        if scaler_method == 'standard':
            scaler = StandardScaler() 
        elif scaler_method == 'minmax':
            scaler = MinMaxScaler() 
        else: # 'none'
            return X, y

        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled_df, y

    def apply_feature_selection(self, X: pd.DataFrame, y: pd.Series, feature_method: str = 'none') -> pd.DataFrame:
        """Aplica RFECV (con persistencia) o PCA (Decisión 3)."""
        print(f"Aplicando selección de características: {feature_method}")

        if feature_method == 'rfecv':
            # Lógica para cargar/guardar el modelo RFECV
            if os.path.exists(RFECV_MODEL_PATH):
                print(f"Cargando modelo RFECV pre-entrenado desde {RFECV_MODEL_PATH}...")
                rfecv = load(RFECV_MODEL_PATH)
            else:
                print("El modelo RFECV no existe. Iniciando entrenamiento (puede tardar)...")
                estimator = LogisticRegression(random_state=self.seed, max_iter=6000, solver='liblinear')
                rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='roc_auc', n_jobs=-1)
                rfecv.fit(X, y)
                dump(rfecv, RFECV_MODEL_PATH) # Guardamos el modelo
                print(f"Entrenamiento completado y modelo guardado en {RFECV_MODEL_PATH}.")
            
            X_reduced = X.loc[:, rfecv.support_]
            print(f"RFECV seleccionó {rfecv.n_features_} características óptimas.")
            return X_reduced
        
        elif feature_method == 'pca':
            # PCA (Análisis de Componentes Principales)
            pca = PCA(n_components=0.95, random_state=self.seed) 
            X_pca = pca.fit_transform(X)
            X_reduced = pd.DataFrame(X_pca, index=X.index)
            
            # El shape[6] indica el número de columnas (características) [7]
            print(f"PCA redujo las dimensiones de {X.shape} a {X_reduced.shape}.")
            return X_reduced

        else: # 'none'
            return X
