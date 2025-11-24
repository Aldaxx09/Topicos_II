# module_data.py
# modulos de terceros
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import os

# modulos propios
from module_path import train_data_path, test_data_path 

COL_TARGET = "DiagPeriodL90D"
RFECV_MODEL_PATH = "rfecv_model.joblib"

class DataProcessor:
    """
    Clase que gestiona la carga, Feature Engineering y preprocesamiento de datos.
    Asegura la integridad metodológica (OOP y Data Leakage prevention).
    """
    def __init__(self, seed: int = 42, test_size: float = 0.2):
        self.seed = seed
        self.test_size = test_size
        # Almacenamos transformadores para la inferencia
        self.numeric_imputer = None
        self.scaler = None
        self.freq_map_zip = None 
        self.selector = None
        self.final_columns = None # CRÍTICO: Guarda el orden de columnas final

    def _row_wise_feature_engineering(self, df):
        """Aplica transformaciones que no dependen de estadísticas globales."""
        df = df.copy()

        # 1. EDAD (Segmentación detallada)
        df['age_18_29'] = np.where((df['patient_age']>=18) & (df['patient_age']<30), 1, 0)
        df['age_30_44'] = np.where((df['patient_age']>=30) & (df['patient_age']<45), 1, 0)
        df['age_45_59'] = np.where((df['patient_age']>=45) & (df['patient_age']<60), 1, 0)
        df['age_60_74'] = np.where((df['patient_age']>=60) & (df['patient_age']<75), 1, 0)
        df['age_75_plus'] = np.where(df['patient_age']>=75, 1, 0)
        df.drop(columns=['patient_age'], inplace=True)

        # 2. ÍNDICE SINTÉTICO DE CONTAMINACIÓN
        cols_contaminacion = ['Ozone', 'PM25', 'N02']
        temp_cont = df[cols_contaminacion].fillna(0)
        df['indice_contaminacion'] = temp_cont.mean(axis=1)

        # 3. BMI (Mapeo Ordinal)
        bins_bmi = [0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf]
        labels_bmi = ['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Extreme']
        df['bmi_category'] = pd.cut(df['bmi'], bins=bins_bmi, labels=labels_bmi, right=False).astype('object')
        df['bmi_category'] = df['bmi_category'].fillna('Unknown')
        
        mapa_bmi = {
            'Unknown': -1, 'Underweight': 0, 'Normal': 1,
            'Overweight': 2, 'Obesity I': 3, 'Obesity II': 4, 'Extreme': 5
        }
        df['bmi_category_enc'] = df['bmi_category'].map(mapa_bmi)
        df.drop(columns=['bmi', 'bmi_category'], inplace=True)

        # 4. PREPARACIÓN ZIP
        df['zip_region'] = df['patient_zip3'].astype(str).str[:3]
        df.drop(columns=['patient_zip3'], inplace=True)
        
        # 5. LIMPIEZA GENERAL (Eliminar redundantes/IDs)
        cols_to_drop = ['patient_id', 'breast_cancer_diagnosis_desc', 
                        'metastatic_first_novel_treatment', 'metastatic_first_novel_treatment_type',
                        'patient_gender']
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        df = df.drop_duplicates()

        return df

    def get_processed_data(self, clean_method='mean', scaler_method='standard', feature_method='rfecv'):
        """Orquesta el FE, Split, Imputación, Escalado y Selección para el entrenamiento."""
        # 1. CARGA
        df_raw = pd.read_csv(train_data_path())

        # 2. FE NIVEL FILA
        df = self._row_wise_feature_engineering(df_raw)

        # 3. SEPARAR TARGET
        y = df[COL_TARGET]
        X = df.drop(columns=[COL_TARGET])

        # 4. SPLIT (CRÍTICO: ANTES de transformaciones estadísticas)
        print("Dividiendo datos (Train/Test) para validación interna...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )

        # 5. FREQUENCY ENCODING (Zip Region) - FIT en Train, TRANSFORM en Test
        self.freq_map_zip = X_train['zip_region'].value_counts()
        X_train['zip_region_freq'] = X_train['zip_region'].map(self.freq_map_zip)
        X_test['zip_region_freq'] = X_test['zip_region'].map(self.freq_map_zip).fillna(0)
        
        X_train.drop(columns=['zip_region'], inplace=True)
        X_test.drop(columns=['zip_region'], inplace=True)

        # 6. ONE-HOT ENCODING
        cat_cols = X_train.select_dtypes(include=['object']).columns
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=float) # Usar float

        X_train_ohe = ohe.fit_transform(X_train[cat_cols])
        X_test_ohe = ohe.transform(X_test[cat_cols])
        
        feat_names = ohe.get_feature_names_out(cat_cols)
        X_train_cat = pd.DataFrame(X_train_ohe, columns=feat_names, index=X_train.index)
        X_test_cat = pd.DataFrame(X_test_ohe, columns=feat_names, index=X_test.index)
        
        X_train_num = X_train.drop(columns=cat_cols)
        X_test_num = X_test.drop(columns=cat_cols)
        
        X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
        X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)
        
        # FIX CRÍTICO DE DTYPE: Convertir a float antes de escalar para eliminar FutureWarnings
        X_train_final = X_train_final.astype(float)
        X_test_final = X_test_final.astype(float)

        # 7. IMPUTACIÓN FINAL (FIT en Train, TRANSFORM en Test)
        if clean_method in ['mean', 'most_frequent']:
            self.numeric_imputer = SimpleImputer(strategy='mean')
            X_train_final[:] = self.numeric_imputer.fit_transform(X_train_final)
            X_test_final[:] = self.numeric_imputer.transform(X_test_final)

        # 8. ESCALADO (FIT en Train, TRANSFORM en Test)
        if scaler_method != 'none':
            self.scaler = StandardScaler() if scaler_method == 'standard' else MinMaxScaler()
            X_train_final[:] = self.scaler.fit_transform(X_train_final)
            X_test_final[:] = self.scaler.transform(X_test_final)

        # 9. SELECCIÓN DE CARACTERÍSTICAS (FIT en Train, TRANSFORM en Test)
        if feature_method == 'rfecv':
            print("Ejecutando RFECV...")
            if os.path.exists(RFECV_MODEL_PATH):
                self.selector = load(RFECV_MODEL_PATH)
            else:
                est = LogisticRegression(random_state=self.seed, max_iter=2000, solver='liblinear')
                self.selector = RFECV(estimator=est, step=1, cv=3, scoring='roc_auc', n_jobs=-1)
                self.selector.fit(X_train_final, y_train)
                dump(self.selector, RFECV_MODEL_PATH)
            
            cols_sel = X_train_final.columns[self.selector.support_]
            X_train_final = X_train_final[cols_sel]
            X_test_final = X_test_final[cols_sel]

        # 11. ALMACENAR COLUMNAS FINALES para Inferencia Externa
        self.final_columns = X_train_final.columns.tolist()

        return X_train_final, X_test_final, y_train, y_test
    
    # --- MÉTODO PARA PROCESAR test.csv (PRÁCTICA) ---

    def process_external_test_data(self):
        """
        Carga el archivo 'test.csv' y aplica las transformaciones APRENDIDAS 
        en el set de entrenamiento (Validación Externa).
        """
        if self.final_columns is None:
            raise Exception("El procesador no ha sido entrenado. Ejecuta get_processed_data primero.")
        
        path = test_data_path()
        df_raw = pd.read_csv(path)
        
        # Guardar IDs
        patient_ids = df_raw['patient_id']
        
        # 1. FE Nivel Fila
        df = self._row_wise_feature_engineering(df_raw)
        
        # 2. Frequency Encoding (APLICAR TRANSFORM)
        df['zip_region_freq'] = df['zip_region'].map(self.freq_map_zip).fillna(0)
        df.drop(columns=['zip_region'], inplace=True)
        
        # 3. ONE-HOT + ALINEACIÓN (CRUCIAL)
        df_aligned_ohe = pd.get_dummies(df, dtype=float)

        # Crea un DataFrame vacío con las columnas del entrenamiento
        X_aligned = pd.DataFrame(0.0, index=df_aligned_ohe.index, columns=self.final_columns)
        
        # Rellenar con los valores que sí existen (la intersección)
        common_cols = df_aligned_ohe.columns.intersection(self.final_columns)
        X_aligned[common_cols] = df_aligned_ohe[common_cols]
        
        # 4. IMPUTACIÓN (APLICAR TRANSFORM)
        X_aligned[:] = self.numeric_imputer.transform(X_aligned)
        
        # 5. ESCALADO (APLICAR TRANSFORM)
        if self.scaler:
            X_aligned[:] = self.scaler.transform(X_aligned)
            
        # 6. SELECCIÓN (APLICAR TRANSFORM)
        if self.selector:
            cols_sel = X_aligned.columns[self.selector.support_]
            X_aligned = X_aligned[cols_sel]
            
        return X_aligned, patient_ids