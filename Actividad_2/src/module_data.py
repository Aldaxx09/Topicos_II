# module_data.py

# Modulos de terceros
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import os

# Modulos propios
from module_path import train_data_path

COL_TARGET = "DiagPeriodL90D"
RFECV_MODEL_PATH = "rfecv_model.joblib"

class DataProcessor:
    def __init__(self, seed: int = 42, test_size: float = 0.2):
        self.seed = seed
        self.test_size = test_size

        # Objetos que se reutilizan en inference
        self.numeric_imputer = None
        self.cat_imputer = None
        self.scaler = None
        self.ohe = None
        self.freq_map_zip = None
        self.selector = None

    def _row_wise_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transformaciones seguras a nivel de fila."""
        df = df.copy()

        # Agrupar la edad en rangos
        df['age_18_29'] = np.where((df['patient_age']>=18) & (df['patient_age']<30), 1, 0)
        df['age_30_44'] = np.where((df['patient_age']>=30) & (df['patient_age']<45), 1, 0)
        df['age_45_59'] = np.where((df['patient_age']>=45) & (df['patient_age']<60), 1, 0)
        df['age_60_74'] = np.where((df['patient_age']>=60) & (df['patient_age']<75), 1, 0)
        df['age_75_plus'] = np.where(df['patient_age']>=75, 1, 0)
        df.drop(columns=['patient_age'], inplace=True)

        # Nueva columna de contaminación promedio
        cols_contaminacion = ['Ozone', 'PM25', 'N02']
        temp_cont = df[cols_contaminacion].fillna(0)
        df['indice_contaminacion'] = temp_cont.mean(axis=1)

        # BMI categorizado y codificado
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

        # ZIP region (primeros 3 dígitos)
        df['zip_region'] = df['patient_zip3'].astype(str).str[:3]
        df.drop(columns=['patient_zip3'], inplace=True)

        # Limpieza general
        df.drop(columns=['patient_gender', 'patient_id', 'breast_cancer_diagnosis_desc'], errors='ignore', inplace=True)
        df = df.drop_duplicates()

        return df

    def get_processed_data(self, clean_method='mean', scaler_method='standard', feature_method='rfecv'):
        # 1. CARGA
        path = train_data_path()
        df_raw = pd.read_csv(path)

        # 2. FE NIVEL FILA
        df = self._row_wise_feature_engineering(df_raw)

        # 3. SEPARAR TARGET
        if COL_TARGET in df.columns:
            y = df[COL_TARGET]
            X = df.drop(columns=[COL_TARGET])
        else:
            raise ValueError(f"Target {COL_TARGET} no encontrado")

        # 4. SPLIT
        print("Dividiendo datos (Train/Test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )

        # 5. FREQUENCY ENCODING (Zip Region)
        # Guardamos el mapa para inference
        self.freq_map_zip = X_train['zip_region'].value_counts()
        X_train['zip_region_freq'] = X_train['zip_region'].map(self.freq_map_zip)
        X_test['zip_region_freq'] = X_test['zip_region'].map(self.freq_map_zip).fillna(0)
        X_train.drop(columns=['zip_region'], inplace=True)
        X_test.drop(columns=['zip_region'], inplace=True)

        # 6. ONE-HOT ENCODING (solo columnas object)
        cat_cols = X_train.select_dtypes(include=['object']).columns

        # Guardamos el encoder para inference
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)

        X_train_ohe = self.ohe.fit_transform(X_train[cat_cols]) if len(cat_cols) > 0 else np.empty((len(X_train), 0))
        X_test_ohe = self.ohe.transform(X_test[cat_cols]) if len(cat_cols) > 0 else np.empty((len(X_test), 0))

        feat_names = self.ohe.get_feature_names_out(cat_cols) if len(cat_cols) > 0 else []
        X_train_cat = pd.DataFrame(X_train_ohe, columns=feat_names, index=X_train.index)
        X_test_cat = pd.DataFrame(X_test_ohe, columns=feat_names, index=X_test.index)

        X_train_num = X_train.drop(columns=cat_cols)
        X_test_num = X_test.drop(columns=cat_cols)

        X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
        X_test_final  = pd.concat([X_test_num, X_test_cat], axis=1)

        # Forzar tipo float para escalado posterior
        X_train_final = X_train_final.astype(float)
        X_test_final  = X_test_final.astype(float)

        # 7. IMPUTACIÓN FINAL (numeric vs. categorical ya están ohe'd -> todo es numérico)
        # Usamos SimpleImputer sobre todo el frame (numérico), evitando None
        if clean_method == 'mean':
            self.numeric_imputer = SimpleImputer(strategy='mean')
        elif clean_method == 'most_frequent':
            self.numeric_imputer = SimpleImputer(strategy='most_frequent')
        else:
            # Fallback seguro
            self.numeric_imputer = SimpleImputer(strategy='mean')

        X_train_final[:] = self.numeric_imputer.fit_transform(X_train_final)
        X_test_final[:]  = self.numeric_imputer.transform(X_test_final)

        # 8. ESCALADO
        if scaler_method == 'standard':
            self.scaler = StandardScaler()
        elif scaler_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_method == 'none':
            self.scaler = None
        else:
            # Fallback seguro
            self.scaler = StandardScaler()

        if self.scaler is not None:
            X_train_final[:] = self.scaler.fit_transform(X_train_final)
            X_test_final[:]  = self.scaler.transform(X_test_final)

        # 9. SELECCIÓN DE CARACTERÍSTICAS
        if feature_method == 'rfecv':
            print("Ejecutando RFECV...")
            if os.path.exists(RFECV_MODEL_PATH):
                # Reutilizamos selector guardado para reproducibilidad
                self.selector = load(RFECV_MODEL_PATH)
            else:
                # class_weight balanced para mejorar recall de la clase minoritaria
                est = LogisticRegression(
                    random_state=self.seed,
                    max_iter=1000,
                    solver='liblinear',
                    class_weight='balanced'
                )
                self.selector = RFECV(estimator=est, step=1, cv=3, scoring='roc_auc', n_jobs=-1)
                self.selector.fit(X_train_final, y_train)
                dump(self.selector, RFECV_MODEL_PATH)

            cols_sel = X_train_final.columns[self.selector.support_]
            X_train_final = X_train_final[cols_sel]
            X_test_final  = X_test_final[cols_sel]
            print(f"\nDatos Listos. Features finales: {len(cols_sel)}")

        return X_train_final, X_test_final, y_train, y_test