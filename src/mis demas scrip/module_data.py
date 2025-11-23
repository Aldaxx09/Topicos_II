# module_data.py

import pandas as pd
from typing import Tuple, List  # <--- CORRECCIÓN: Importamos Tuple y List

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer  # <--- CORRECCIÓN: Importamos ColumnTransformer
from sklearn.pipeline import Pipeline

# Módulos propios
from module_path import train_data_path

# --- Columnas del Dataset (basadas en tu archivo) ---
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
COL_NEVER_MARRIED = "never_married"
COL_DIVORCED = "divorced"
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
COL_DIAGPERIODL90D = "DiagPeriodL90D" # variable objetivo

# --- Columnas Clave ---
TARGET_COLUMN = COL_DIAGPERIODL90D

# Constantes
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

# --- Definición de Features (Basado en las constantes de arriba) ---
CATEGORICAL_FEATURES = [
    COL_PATIENT_RACE, COL_PAYER_TYPE, COL_PATIENT_STATE, COL_PATIENT_ZIP3,
    COL_PATIENT_GENDER, COL_BREAST_CANCER_DIAGNOSIS_CODE,
    COL_BREAST_CANCER_DIAGNOSIS_DESC, COL_METASTATIC_CANCER_DIAGNOSIS_CODE,
    COL_METASTATIC_FIRST_NOVEL_TREATMENT, COL_METASTATIC_FIRST_NOVEL_TREATMENT_TYPE,
    COL_REGION, COL_DIVISION
]

NUMERIC_FEATURES = [
    COL_PATIENT_AGE, COL_BMI, COL_POPULATION, COL_DENSITY, COL_AGE_MEDIAN,
    COL_AGE_UNDER_10, COL_AGE_10_TO_19, COL_AGE_20S, COL_AGE_30S, COL_AGE_40S,
    COL_AGE_50S, COL_AGE_60S, COL_AGE_70S, COL_AGE_OVER_80, COL_MALE, COL_FEMALE,
    COL_MARRIED, COL_NEVER_MARRIED, COL_DIVORCED, COL_WIDOWED, COL_FAMILY_SIZE,
    COL_FAMILY_DUAL_INCOME, COL_INCOME_HOUSEHOLD_MEDIAN,
    COL_INCOME_HOUSEHOLD_UNDER_5, COL_INCOME_HOUSEHOLD_5_TO_10,
    COL_INCOME_HOUSEHOLD_10_TO_15,
    COL_INCOME_HOUSEHOLD_15_TO_20,  
    COL_INCOME_HOUSEHOLD_20_TO_25, 
    COL_INCOME_HOUSEHOLD_25_TO_35,
    COL_INCOME_HOUSEHOLD_35_TO_50, COL_INCOME_HOUSEHOLD_50_TO_75,
    COL_INCOME_HOUSEHOLD_75_TO_100, COL_INCOME_HOUSEHOLD_100_TO_150,
    COL_INCOME_HOUSEHOLD_150_OVER, COL_INCOME_HOUSEHOLD_SIX_FIGURE,
    COL_INCOME_INDIVIDUAL_MEDIAN, COL_HOME_OWNERSHIP, COL_HOUSING_UNITS,
    COL_HOME_VALUE, COL_RENT_MEDIAN, COL_RENT_BURDEN,
    COL_EDUCATION_LESS_HIGHSCHOOL, COL_EDUCATION_HIGHSCHOOL,
    COL_EDUCATION_SOME_COLLEGE, COL_EDUCATION_BACHELORS,
    COL_EDUCATION_GRADUATE, COL_EDUCATION_COLLEGE_OR_ABOVE,
    COL_EDUCATION_STEM_DEGREE, COL_LABOR_FORCE_PARTICIPATION,
    COL_UNEMPLOYMENT_RATE, COL_SELF_EMPLOYED, COL_FARMER, COL_RACE_WHITE,
    COL_RACE_BLACK, COL_RACE_ASIAN, COL_RACE_NATIVE, COL_RACE_PACIFIC,
    COL_RACE_OTHER, COL_RACE_MULTIPLE, COL_HISPANIC, COL_DISABLED,
    COL_POVERTY, COL_LIMITED_ENGLISH, COL_COMMUTE_TIME,
    COL_HEALTH_UNINSURED, COL_VETERAN
]

def load_data() -> pd.DataFrame:
    """
    Carga los datos de entrenamiento desde el path definido.
    """
    path = train_data_path()
    print(f"Cargando datos desde: {path}")
    return pd.read_csv(path)

def get_feature_definitions(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Define las features numéricas y categóricas basado en Wids2024.ipynb.
    """
    # Filtramos para asegurar que solo usamos columnas que existen en el DataFrame
    existing_numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    existing_categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    
    print(f"Features numéricas encontradas: {len(existing_numeric)}")
    print(f"Features categóricas encontradas: {len(existing_categorical)}")

    return existing_numeric, existing_categorical

def create_preprocessor(
    numeric_features: List[str], 
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Crea el ColumnTransformer basado en la lógica de Wids2024.ipynb.
    """
    print("Creando pipeline de preprocesamiento (ColumnTransformer)...")
    
    # Pipeline para features numéricas (de tu notebook)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())                 
    ])
    
    # Pipeline para features categóricas (de tu notebook)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore'))   
    ])
    
    # Ensamblamos con ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Ignoramos columnas no especificadas (como 'patient_id')
    )
    return preprocessor

def get_train_test_split(df: pd.DataFrame) -> Tuple:
    """
    Divide el DataFrame en conjuntos de entrenamiento y prueba.
    """
    print("Dividiendo datos en train/test split...")
    
    # Asegurarnos que la columna objetivo existe
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COLUMN}' no se encontró en los datos.")
        
    y = df[TARGET_COLUMN]
    X = df.drop(TARGET_COLUMN, axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_SEED,
        stratify=y # Importante para clasificación desbalanceada
    )
    return X_train, X_test, y_train, y_test