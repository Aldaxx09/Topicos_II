# Importar librerias estandar
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Scikit-learn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Importar modulos propios
from module_path2 import test_data_path, train_data_path

COL_PATIENT_ID = "patient_id"
COL_PATIENT_RACE = "patient_race"
COL_PAYER_TYPE = "payer_type"
COL_PATIENT_STATE = "patient_state"
COL_PATIENT_ZIP3 = "patient_zip3"
COL_PATIENT_AGE = "patient_age"
COL_PATIENT_GENDER = "patient_gender"
COL_BMI = "bmi"
COL_BREAST_CANCER_DIAGNOSIS_CODE = "breast_cancer_diagnosis_code"
COL_BREAST_CANCER_DIAGNOSIS_DESC = "breast_cancer_diagnosis_desc"
COL_METASTATIC_CANCER_DIAGNOSIS_CODE = "metastatic_cancer_diagnosis_code"
COL_METASTATIC_FIRST_NOVEL_TREATMENT = "metastatic_first_novel_treatment"
COL_METASTATIC_FIRST_NOVEL_TREATMENT_TYPE = "metastatic_first_novel_treatment_type"
COL_REGION = "Region"
COL_DIVISION = "Division"
COL_POPULATION = "population"
COL_DENSITY = "density"
COL_AGE_MEDIAN = "age_median"
COL_AGE_UNDER_10 = "age_under_10"
COL_AGE_10_TO_19 = "age_10_to_19"
COL_AGE_20S = "age_20s"
COL_AGE_30S = "age_30s"
COL_AGE_40S = "age_40s"
COL_AGE_50S = "age_50s"
COL_AGE_60S = "age_60s"
COL_AGE_70S = "age_70s"
COL_AGE_OVER_80 = "age_over_80"
COL_MALE = "male"
COL_FEMALE = "female"
COL_MARRIED = "married"
COL_DIVORCED = "divorced"
COL_NEVER_MARRIED = "never_married"
COL_WIDOWED = "widowed"
COL_FAMILY_SIZE = "family_size"
COL_FAMILY_DUAL_INCOME = "family_dual_income"
COL_INCOME_HOUSEHOLD_MEDIAN = "income_household_median"
COL_INCOME_HOUSEHOLD_UNDER_5 = "income_household_under_5"
COL_INCOME_HOUSEHOLD_5_TO_10 = "income_household_5_to_10"
COL_INCOME_HOUSEHOLD_10_TO_15 = "income_household_10_to_15"
COL_INCOME_HOUSEHOLD_15_TO_20 = "income_household_15_to_20"
COL_INCOME_HOUSEHOLD_20_TO_25 = "income_household_20_to_25"
COL_INCOME_HOUSEHOLD_25_TO_35 = "income_household_25_to_35"
COL_INCOME_HOUSEHOLD_35_TO_50 = "income_household_35_to_50"
COL_INCOME_HOUSEHOLD_50_TO_75 = "income_household_50_to_75"
COL_INCOME_HOUSEHOLD_75_TO_100 = "income_household_75_to_100"
COL_INCOME_HOUSEHOLD_100_TO_150 = "income_household_100_to_150"
COL_INCOME_HOUSEHOLD_150_OVER = "income_household_150_over"
COL_INCOME_HOUSEHOLD_SIX_FIGURE = "income_household_six_figure"
COL_INCOME_INDIVIDUAL_MEDIAN = "income_individual_median"
COL_HOME_OWNERSHIP = "home_ownership"
COL_HOUSING_UNITS = "housing_units"
COL_HOME_VALUE = "home_value"
COL_RENT_MEDIAN = "rent_median"
COL_RENT_BURDEN = "rent_burden"
COL_EDUCATION_LESS_HIGHSCHOOL = "education_less_highschool"
COL_EDUCATION_HIGHSCHOOL = "education_highschool"
COL_EDUCATION_SOME_COLLEGE = "education_some_college"
COL_EDUCATION_BACHELORS = "education_bachelors"
COL_EDUCATION_GRADUATE = "education_graduate"
COL_EDUCATION_COLLEGE_OR_ABOVE = "education_college_or_above"
COL_EDUCATION_STEM_DEGREE = "education_stem_degree"
COL_LABOR_FORCE_PARTICIPATION = "labor_force_participation"
COL_UNEMPLOYMENT_RATE = "unemployment_rate"
COL_SELF_EMPLOYED = "self_employed"
COL_FARMER = "farmer"
COL_RACE_WHITE = "race_white"
COL_RACE_BLACK = "race_black"
COL_RACE_ASIAN = "race_asian"
COL_RACE_NATIVE = "race_native"
COL_RACE_PACIFIC = "race_pacific"
COL_RACE_OTHER = "race_other"
COL_RACE_MULTIPLE = "race_multiple"
COL_HISPANIC = "hispanic"
COL_DISABLED = "disabled"
COL_POVERTY = "poverty"
COL_LIMITED_ENGLISH = "limited_english"
COL_COMMUTE_TIME = "commute_time"
COL_HEALTH_UNINSURED = "health_uninsured"
COL_VETERAN = "veteran"
COL_OZONE = "Ozone"
COL_PM25 = "PM25"
COL_N02 = "N02"
COL_DIAGPERIODL90D = "DiagPeriodL90D"  # variable objetivo



TARGET_COLUMN = COL_DIAGPERIODL90D  # Define cuál columna es la que queremos predecir
TEST_SPLIT_SIZE = 0.2  # Usaremos un 20% de los datos para el conjunto de prueba
RANDOM_SEED = 42       # Una semilla fija para que la división sea reproducible



class Dataset():
    
    def __init__(self, num_samples:int=None, seed:int=42):
        self.num_samples = num_samples
        self.seed = seed
        
        
        
    # Cargar datos desde un archivo CSV
    def load_data(self):
        # Ruta de archivos
        train_path = train_data_path()
        test_path  = test_data_path()

        # Leer con pd
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    
        # Lista de columnas inncesarias
        cols=['patient_id','breast_cancer_diagnosis_desc',
                 'metastatic_first_novel_treatment',
                 'metastatic_first_novel_treatment_type']
        # las elimina para train y test
        df_train = df_train.drop(columns=cols)
        df_test = df_test.drop(columns=cols)
        print("Columans eliminadas:", cols)
        
        # Si num_sample no es nulo entonces hara: una muestra pequeña  de train y test
        if self.num_samples is not None:
            df_train = df_train.sample(n=self.num_samples,
                                       random_state=self.seed)
            df_test = df_test.sample(n=self.num_samples,
                                     random_state=self.seed)
        return df_train, df_test


    
    # Metodo para limpiar datos 
    def load_data_clean_1(self):
        df_train, df_test = self.load_data()
        '''
        Usa el metodo de imputacion para limpiar datos nulos en test
        y dropna para limpiar datos nulos en train.
        '''
        # Tratamiento de datos nulos
        # BMI
        df_train['bmi_category'] = pd.cut(df_train['bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9,39.9, np.inf], labels=['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II','Extreme']).astype(object)
        df_test['bmi_category'] = pd.cut(df_test['bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9,39.9, np.inf], labels=['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II','Extreme']).astype(object)
        df_train['bmi_category'] = df_train['bmi_category'].fillna('Unknown')
        df_test['bmi_category'] = df_test['bmi_category'].fillna('Unknown')
        df_train.drop(columns=['bmi'], inplace=True) # Elimina la columna original 'bmi' del DataFrame de entrenamiento.
        df_test.drop(columns=['bmi'], inplace=True) # Elimina la columna original 'bmi' del DataFrame de prueba.
        # patient_race
        df_train['patient_race'] = df_train['patient_race'].fillna('Unknown')# Rellena los valores nulos en la columna 'patient_race' con 'Unknown'.
        df_test['patient_race'] = df_test['patient_race'].fillna('Unknown') # lo mismo para el conjunto de prueba.
        # payer_type
        df_train['payer_type'] = df_train['payer_type'].fillna('No insurance')# Rellena los valores nulos en la columna 'payer_type' con 'No insurance'.
        df_test['payer_type'] = df_test['payer_type'].fillna('No insurance') # lo mismo para el conjunto de prueba.
        # Dropna (solo a training)
        df_train = df_train.dropna()
        
        # Fill test data
        test_categories = df_test.select_dtypes(include=['object']).columns.tolist()# lista cat
        test_numeric = df_test.select_dtypes(include=['number']).columns.tolist()# lista num
        num_imputer = SimpleImputer(strategy='mean') # metodo de imput media 
        df_test[test_numeric] = num_imputer.fit_transform(df_test[test_numeric])
        cat_imputer = SimpleImputer(strategy='most_frequent')# metodo mediana
        df_test[test_categories] = cat_imputer.fit_transform(df_test[test_categories])
        print("Datos nulos limpiados en train con dropna y en test con imputacion.")
        print(f"Valores nulos restantes en train: {df_train.isnull().sum().sum()}")
        print(f"Valores nulos restantes en test: {df_test.isnull().sum().sum()}")
        
        return df_train, df_test

    def load_data_clean_2(self):
        df_train, df_test = self.load_data()
        # Tratamiento de datos nulos
        '''
        Elimina filas con datos nulos en train y test sin imputacion.
        '''
        # BMI
        df_train['bmi_category'] = pd.cut(df_train['bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9,39.9, np.inf], labels=['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II','Extreme']).astype(object)
        df_test['bmi_category'] = pd.cut(df_test['bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9,39.9, np.inf], labels=['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II','Extreme']).astype(object)
        df_train['bmi_category'] = df_train['bmi_category'].fillna('Unknown')
        df_test['bmi_category'] = df_test['bmi_category'].fillna('Unknown')
        df_train.drop(columns=['bmi'], inplace=True) # Elimina la columna original 'bmi' del DataFrame de entrenamiento.
        df_test.drop(columns=['bmi'], inplace=True) # Elimina la columna original 'bmi' del DataFrame de prueba.
        # patient_race
        df_train['patient_race'] = df_train['patient_race'].fillna('Unknown')# Rellena los valores nulos en la columna 'patient_race' con 'Unknown'.
        df_test['patient_race'] = df_test['patient_race'].fillna('Unknown') # lo mismo para el conjunto de prueba.
        # payer_type
        df_train['payer_type'] = df_train['payer_type'].fillna('No insurance')# Rellena los valores nulos en la columna 'payer_type' con 'No insurance'.
        df_test['payer_type'] = df_test['payer_type'].fillna('No insurance') # lo mismo para el conjunto de prueba.
        # Dropna (solo a training)
        df_train = df_train.dropna()
        df_test = df_test.dropna()
        print(f"Valores nulos restantes en train: {df_train.isnull().sum().sum()}")
        print(f"Valores nulos restantes en test: {df_test.isnull().sum().sum()}")
        print("Datos nulos eliminados en train y test usando dropna.")
        
        return df_train, df_test

    # def load_data_clean_duplicated(self):
    #     print(f"Duplicados en Train: {df_train.duplicated().sum()}")
    #     print(f"Duplicados en Test: {df_test.duplicated().sum()}")
    #     # los eliminaríamos con:
    #     df_train = df_train.drop_duplicates()
    #     df_test = df_test.drop_duplicates()
    #     # Verificamos:
    #     print(f"Duplicados en Train: {df_train.duplicated().sum()}")
    #     print(f"Duplicados en Test: {df_test.duplicated().sum()}")
    #     return df_train, df_test
        
    def load_data_clean_duplicated(self):
        # Asegurarse de tener los DataFrames
        df_train, df_test = self.load_data()

        # Diagnóstico rápido
        print("Shapes:", df_train.shape, df_test.shape)
        print("Duplicados en Train (todas columnas):", df_train.duplicated().sum())
        print("Duplicados en Test (todas columnas):", df_test.duplicated().sum())

        # Mostrar algunos duplicados para inspección
        if df_train.duplicated().any():
            print(df_train[df_train.duplicated(keep=False)].head(10))
        if df_test.duplicated().any():
            print(df_test[df_test.duplicated(keep=False)].head(10))

        # Normalizar columnas de texto si es necesario (ejemplo)
        # for col in ['patient_race', 'payer_type']:
        #     if col in df_train.columns:
        #         df_train[col] = df_train[col].astype(str).str.strip().str.lower()
        #         df_test[col]  = df_test[col].astype(str).str.strip().str.lower()

        # Eliminar duplicados (ejemplo: por todas las columnas)
        df_train = df_train.drop_duplicates()
        df_test  = df_test.drop_duplicates()

        # Opcional: eliminar duplicados basados en un subconjunto relevante
        # df_train = df_train.drop_duplicates(subset=['col1','col2'], keep='first')

        # Resetear índices
        df_train = df_train.reset_index(drop=True)
        df_test  = df_test.reset_index(drop=True)

        print("Duplicados en Train (post):", df_train.duplicated().sum())
        print("Duplicados en Test (post):", df_test.duplicated().sum())

        return df_train, df_test
        
        
    
    def lead_data_clean_3(self):
        df_train, df_test = self.load_data()
        # Tratamiento de datos nulos
        '''
        Imputación por Vecinos Cercanos (KNNlmputer)
        Debe ejecutarse DESPUÉS de la codificación one-hot (ya que solo funciona con números).
        '''
        # BMI
        df_train['bmi_category'] = pd.cut(df_train['bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9,39.9, np.inf], labels=['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II','Extreme']).astype(object)
        df_test['bmi_category'] = pd.cut(df_test['bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9,39.9, np.inf], labels=['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II','Extreme']).astype(object)
        df_train['bmi_category'] = df_train['bmi_category'].fillna('Unknown')
        df_test['bmi_category'] = df_test['bmi_category'].fillna('Unknown')
        df_train.drop(columns=['bmi'], inplace=True) # Elimina la columna original 'bmi' del DataFrame de entrenamiento.
        df_test.drop(columns=['bmi'], inplace=True) # Elimina la columna original 'bmi' del DataFrame de prueba.
        # patient_race
        df_train['patient_race'] = df_train['patient_race'].fillna('Unknown')# Rellena los valores nulos en la columna 'patient_race' con 'Unknown'.
        df_test['patient_race'] = df_test['patient_race'].fillna('Unknown') # lo mismo para el conjunto de prueba.
        # payer_type
        df_train['payer_type'] = df_train['payer_type'].fillna('No insurance')# Rellena los valores nulos en la columna 'payer_type' con 'No insurance'.
        df_test['payer_type'] = df_test['payer_type'].fillna('No insurance') # lo mismo para el conjunto de prueba.
        # Dropna (solo a training)
        
        
        print("Iniciando KNN Imputer")
        knn_imputer = KNNImputer(n_neighbors=5) # n_neighbors=5 es un valor por defecto común
        # 1. Separar X e y de 'df_normalizado' 
        X_normalizado = df_normalizado.drop(columns=['DiagPeriodL90D'])
        y_normalizado = df_normalizado['DiagPeriodL90D']
        # 2. Guardar columnas e índices (porque fit_transform devuelve un array)
        X_cols = X_normalizado.columns
        X_idx = X_normalizado.index
        test_cols = df_test_aligned.columns
        test_idx = df_test_aligned.index
        # 3. Ajustar y transformar en Train (X_normalizado)
        X_imputado_np = knn_imputer.fit_transform(X_normalizado)
        X_normalizado = pd.DataFrame(X_imputado_np, columns=X_cols, index=X_idx)
        # 4. Transformar SÓLO en Test (df_test_aligned)
        df_test_imputado_np = knn_imputer.transform(df_test_aligned)
        df_test_aligned = pd.DataFrame(df_test_imputado_np, columns=test_cols, index=test_idx)
        # 5. Reconstruir 'df_normalizado' (con 'y') para que el resto del script (PCA/RFECV) lo use
        df_normalizado = X_normalizado.copy()
        df_normalizado['DiagPeriodL90D'] = y_normalizado.values # .values es importante para alinear
        print("¡Imputación con KNN completada!")
        print(f"Valores nulos restantes en train: {df_normalizado.isnull().sum().sum()}")
        print(f"Valores nulos restantes en test: {df_test_aligned.isnull().sum().sum()}")
        
        return df_normalizado, df_test_aligned


    def load_data_clean_duplicated(self):
        print(f"Duplicados en Train: {df_training.duplicated().sum()}")
        print(f"Duplicados en Test: {df_test.duplicated().sum()}")
        # los eliminaríamos con:
        df_training = df_training.drop_duplicates()
        df_test = df_test.drop_duplicates()
        # Verificamos:
        print(f"Duplicados en Train: {df_training.duplicated().sum()}")
        print(f"Duplicados en Test: {df_test.duplicated().sum()}")
                
        return df_training, df_test
        

    def load_data_clean_duplicated_encoded(self):
        df_train, df_test = self.load_data_clean_duplicated()
        
        test_categories = df_test.select_dtypes(include=['object']).columns.tolist()

        df_train_encoded = pd.get_dummies(df_train, columns=test_categories, drop_first=False, dtype=int)
        df_test_encoded = pd.get_dummies(df_test, columns=test_categories, drop_first=False, dtype=int)

        return df_train_encoded, df_test_encoded 


if __name__ == "__main__":
    ds = Dataset(num_samples=None, seed=42)
    df_train, df_test = ds.load_data_clean_duplicated()