# Topicos_II

# Proyecto 1 :

## 🚀 WiDS Datathon 2024 - Análisis de Equidad en IA de Salud

Este repositorio contiene un Análisis Exploratorio de Datos (EDA) exhaustivo para el **WiDS Datathon 2024 Challenge 1: Equidad en la IA de Salud**, organizado por Kaggle.

El objetivo principal del desafío es desarrollar modelos de machine learning que puedan predecir diagnósticos médicos asegurando un **rendimiento equitativo** entre diferentes grupos demográficos.

## 📖 Descripción del Proyecto

El notebook `EDA.ipynb` documenta el proceso completo de limpieza de datos, análisis de variables, ingeniería de características y preprocesamiento. El enfoque principal de este análisis no es solo encontrar predictores, sino **identificar y visualizar activamente las disparidades** en las tasas de diagnóstico entre diferentes cohortes de pacientes, sentando las bases para un modelado justo.

### Dataset

Los datos provienen de la competición oficial de Kaggle:

- [Kaggle WiDS Datathon 2024 Challenge 1](https://www.kaggle.com/competitions/widsdatathon2024-challenge1/data)

---

## 📊 Análisis Realizado

El proceso de EDA siguió los siguientes pasos:

⚙️ Metodología (Pipeline de Ciencia de Datos)
El proceso sigue las fases críticas de un proyecto de análisis de datos, asegurando la calidad de los datos y la robustez del modelo:
Fase 1: Análisis Exploratorio de Datos (EDA) y Limpieza
Esta fase es crucial, ya que la calidad de los datos es el principal riesgo en cualquier proyecto de Ciencia de Datos. Los errores no detectados aquí obligan a retroceder en fases avanzadas del proyecto.

1. Revisión y Estructura:
   ◦ Se cargaron los conjuntos de entrenamiento (df_training) y prueba (df_test).
   ◦ Las dimensiones iniciales de df_training fueron de 12,906 filas y 79 columnas, mientras que df_test contenía 5,792 filas y 78 columnas. La diferencia de columnas se debe a la ausencia de la variable objetivo (DiagPeriodL90D) en el conjunto de prueba.
   ◦ Se distinguieron las variables numéricas y categóricas para aplicar transformaciones adecuadas.
2. Manejo de Datos Faltantes (Nulos) e Inconsistencias:
   ◦ Se identificaron variables con alto porcentaje de nulos, como bmi (~69.5%) y patient_race (~49.5%).
   ◦ Estrategia de Imputación: Se utilizó una imputación informativa para los altos porcentajes de nulos, etiquetando los valores faltantes en bmi como 'Unknown' y en patient_race como 'Unknown'. Para el resto de nulos con bajo porcentaje, se empleó la imputación simple con la media (para numéricas) o la moda (para categóricas) [EDA33, 271, 619].
   ◦ Inconsistencias Categóricas: Para asegurar que los datasets de entrenamiento y prueba tuvieran los mismos valores únicos en columnas categóricas, se implementó la concatenación previa a la codificación, evitando un sesgo en las características [EDA33, 262].
3. Variables Irrelevantes y Ruido:
   ◦ Se eliminó la columna patient_gender debido a que era constante (varianza cercana a cero).
   ◦ El análisis de correlación identificó variables con muy baja señal predictiva (ruido probable).
   ◦ Se confirmó la ausencia de registros duplicados en ambos conjuntos.
   Fase 2: Preprocesamiento y Reducción de Dimensionalidad
   El preprocesamiento es una etapa crucial para que los datos puedan ser utilizados por los algoritmos de ML.
4. Transformaciones y Codificación:
   ◦ Codificación de Variables Categóricas: Se aplicó One-Hot Encoding para convertir las variables categóricas (object) en valores numéricos binarios, lo cual es necesario para la mayoría de los modelos [EDA33, 262].
   ◦ Escalado y Estandarización: Se utiliza StandardScaler o MinMaxScaler para asegurar que los atributos estén en la misma escala (media cero y desviación estándar uno), una transformación vital para algoritmos sensibles a la distancia o al gradiente [cod_final, 420, 440].
5. Selección de Características (Feature Selection):
   ◦ Eliminación Recursiva de Características (RFECV): Se implementó RFECV para buscar el subconjunto óptimo de variables predictoras [cod_final, 349]. Reducir el número de variables (dimensionalidad) puede mejorar la eficiencia computacional y el tiempo de respuesta del modelo.
   ◦ Persistencia: El modelo RFECV se guardó (rfecv_model.joblib) para evitar el recálculo y asegurar la reproducibilidad.
6. Análisis de Componentes Principales (PCA):
   ◦ Se exploró el PCA como una técnica de reducción de dimensionalidad para transformar variables correlacionadas en nuevos componentes no correlacionados [EDA33, 332]. La matriz de Cargas Factoriales (Loadings) ayuda a interpretar la contribución de las variables originales a estos nuevos componentes.
   Fase 3: Modelado y Evaluación
   El proceso de modelado utilizó la validación cruzada para obtener una estimación robusta del rendimiento.
7. Comparación de Modelos: Se evaluó un amplio diccionario de clasificadores para la tarea de clasificación [cod_final, EDA33, 360], incluyendo:
   ◦ Regresión Logística
   ◦ Naive Bayes (modelo probabilístico)
   ◦ Árbol de Decisión (modelo interpretable)
   ◦ Random Forest y AdaBoost (modelos de ensemble)
   ◦ k-Nearest Neighbors (k-NN) (modelo basado en distancia)
   ◦ SGD Classifier
   ◦ LightGBM
   ◦ MLP (Multi-Layer Perceptron / Red Neuronal)
8. Métrica de Evaluación:
   ◦ La métrica principal utilizada para la selección de modelos fue el AUROC (roc_auc), ya que es fundamental para evaluar modelos de clasificación [cod_final, 592, 601].
   ◦ El rendimiento también se detalló mediante el Reporte de Clasificación y la Matriz de Confusión (analizando Verdaderos/Falsos Positivos y Negativos) [cod_final, 600].
9. Modelo Seleccionado (Actual):
   ◦ El modelo_final designado en el código es la Regresión Logística.
   ◦ Curva ROC: La evaluación de la curva ROC del modelo final (Regresión Logística) en los conjuntos de entrenamiento y prueba mostró que el modelo no tiene sobreajuste (overfitting).

## Conclusiones y Tareas Futuras

Hallazgos Clave

• La variable objetivo (DiagPeriodL90D) presenta desbalance de clases

• El conjunto de datos requirió un esfuerzo significativo de limpieza y preprocesamiento para manejar los valores nulos.

• Imputación: El tratamiento de valores nulos, especialmente en bmi, mediante la creación de categorías Unknown, resultó ser una característica predictiva útil.

• Proceso Estructurado: El uso de RFECV para la selección de características y la validación cruzada (cross_validate) garantizan un proceso riguroso y una métrica de rendimiento confiable, evitando errores metodológicos [cod_final, 257, 345].
Reflexiones sobre Implementación (IA/Scheme)

• Se detectaron y eliminaron columnas sin valor predictivo (como patient_gender)

• El análisis de correlación y la importancia de características mediante Random Forest identificaron variables candidatas a ser eliminadas por ser ruido probable.

## 🛠️ Tecnologías Utilizadas

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (para `PCA`, `StandardScaler`, `SimpleImputer`)
- Jupyter Notebook

## ▶️ Cómo Ejecutar este Proyecto

1.  Clona este repositorio:
    ```bash
    git clone https://github.com/Aldaxx09/Topicos_II.git
    cd [Temas_II]
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
