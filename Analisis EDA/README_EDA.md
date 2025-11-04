# Topicos_II

# Proyecto 1:
## 🚀 WiDS Datathon 2024 - Análisis de Equidad en IA de Salud

Este repositorio contiene un Análisis Exploratorio de Datos (EDA) exhaustivo para el **WiDS Datathon 2024 Challenge 1: Equidad en la IA de Salud**, organizado por Kaggle.

El objetivo principal del desafío es desarrollar modelos de machine learning que puedan predecir diagnósticos médicos asegurando un **rendimiento equitativo** entre diferentes grupos demográficos.

## 📖 Descripción del Proyecto

El notebook `EDA.ipynb` documenta el proceso completo de limpieza de datos, análisis de variables, ingeniería de características y preprocesamiento. El enfoque principal de este análisis no es solo encontrar predictores, sino **identificar y visualizar activamente las disparidades** en las tasas de diagnóstico entre diferentes cohortes de pacientes, sentando las bases para un modelado justo.

### Dataset

Los datos provienen de la competición oficial de Kaggle:
* [Kaggle WiDS Datathon 2024 Challenge 1](https://www.kaggle.com/competitions/widsdatathon2024-challenge1/data)

---

## 📊 Análisis Realizado

El proceso de EDA siguió los siguientes pasos:

1.  **Carga y Limpieza Inicial:**
    * Carga de los archivos `training.csv` y `test.csv`.
    * Eliminación de columnas irrelevantes (`patient_id`, `breast_cancer_diagnosis_desc`, etc.).

2.  **Manejo de Valores Nulos:**
    * Imputación estratégica para las variables con más nulos (`bmi`, `patient_race`, `payer_type`).
    * `bmi` fue convertido a una variable categórica (`bmi_category`) para manejar sus nulos.
    * `patient_race` y `payer_type` se rellenaron con categorías 'Unknown' y 'No insurance' respectivamente.
    * Eliminación de filas con valores nulos restantes (representando un bajo porcentaje del total).

3.  **Análisis de Equidad (Núcleo del Desafío):**
    * Análisis de la **tasa de diagnóstico** (`DiagPeriodL90D == 1`) a través de grupos demográficos clave.
    * Visualización de proporciones para `patient_race`, `payer_type` y `Region` para detectar sesgos.

4.  **Análisis Exploratorio (EDA):**
    * **Análisis Univariado:** Distribución de la variable objetivo (desbalance de clases) y de predictores numéricos.
    * **Análisis Bivariado:** Comparación de las distribuciones de variables (`patient_age`, `income_household_median`, `poverty`) contra la variable objetivo usando `kdeplot` y `boxplot` con `hue`.
    * **Análisis de Outliers:** Identificación de valores atípicos mediante Boxplots y QQ-Plots.

5.  **Análisis de Características y Correlación:**
    * Mapa de calor para correlaciones entre variables numéricas.
    * Prueba de **Chi-Cuadrado** para identificar la asociación entre variables categóricas (ej. `Region` y `Division`).

6.  **Preparación para Modelado (Preprocesamiento):**
    * **Codificación:** Aplicación de *One-Hot Encoding* (`pd.get_dummies`) a todas las variables categóricas.
    * **Alineación de Columnas:** Se aseguró que los dataframes de *train* y *test* tuvieran exactamente las mismas columnas después del encoding.
    * **Normalización:** Escalado de todas las características usando `StandardScaler`.
    * **Reducción de Dimensionalidad:** Aplicación de **PCA** (Análisis de Componentes Principales) para reducir el número de características, reteniendo el 95% de la varianza.

---

## 💡 Hallazgos Clave

* **Desbalance de Clases:** El dataset está **fuertemente desbalanceado**. La clase positiva (`DiagPeriodL90D == 1`) representa  el **~67%** del total.
* **Disparidad en Diagnóstico:** El análisis de equidad confirmó **diferencias medibles** en la tasa de diagnóstico positivo entre diferentes grupos raciales y tipos de seguro médico. Esto valida el enfoque del desafío en la *equidad*.
* **Predictores Potenciales:** Las variables demográficas (`patient_age`) y socioeconómicas (`income_household_median`, `poverty`) mostraron una mayor separación visual con la variable objetivo que las variables ambientales (`Ozone`, `PM25`).
* **Reducción de Dimensionalidad:** El dataset original, tras la codificación one-hot, superó las 200 columnas. PCA fue efectivo para reducirlo a **~156 componentes** (o el número que te haya dado) explicando el 95% de la varianza.

---

## 🛠️ Tecnologías Utilizadas

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn (para `PCA`, `StandardScaler`, `SimpleImputer`)
* Jupyter Notebook

---

## ▶️ Cómo Ejecutar este Proyecto

1.  Clona este repositorio:
    ```bash
    git clone [https://github.com/](https://github.com/)[TuUsuario]/[TuRepositorio].git
    cd [TuRepositorio]
    ```
2.  (Recomendado) Crea un entorno virtual:
    ```bash
    python -m venv venv
    source vVenv/bin/activate  # En Windows: venv\Scripts\activate
    ```
3.  Instala las dependencias:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
4.  **Importante:** Descarga los archivos `training.csv` y `test.csv` desde la [página de la competición en Kaggle](https://www.kaggle.com/competitions/widsdatathon2024-challenge1/data) y colócalos en la raíz del repositorio.

5.  Inicia Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
6.  Abre y ejecuta el archivo `[Nombre de tu notebook.ipynb]`.