# 🚀 Pipeline Modular de Machine Learning - WIDS Datathon 2024

> **Actividad 2 - Tópicos Selectos II** > _Enfoque: MLOps, Modularidad y Reproducibilidad._

Este repositorio contiene una solución de arquitectura modular para el desafío **WiDS Datathon 2024: Equidad en la IA de Salud**. El objetivo es predecir diagnósticos médicos (`DiagPeriodL90D`) asegurando un flujo de trabajo robusto, libre de _Data Leakage_ y con seguimiento de experimentos automatizado.

---

## 📖 Descripción del Proyecto

El proyecto transforma un análisis exploratorio inicial (EDA) en un **Pipeline de Producción** siguiendo principios de Diseño Orientado a Objetos (OOP).

### Características Clave:

- **🏗️ Arquitectura Modular:** Separación estricta entre Datos (`DataProcessor`), Modelado (`ModelEvaluator`) y Orquestación (`main`).
- **🛡️ Prevención de Data Leakage:** El preprocesamiento (imputación, escalado, selección) se ajusta (_fit_) estrictamente en el conjunto de entrenamiento y se aplica (_transform_) al conjunto de prueba.
- **🔬 Experimentación Trackeada:** Integración con **MLflow** para registrar métricas, parámetros y artefactos de modelos automáticamente.
- **⚙️ Ingeniería de Características Avanzada:** Segmentación de edad, índice sintético de contaminación, codificación ordinal de BMI y Frequency Encoding de zonas postales.

---

## 📂 Estructura del Repositorio

├── data/ # Conjuntos de datos (training.csv, test.csv)
├── mlruns/ # Registro de experimentos de MLflow
├── src/ # Código fuente modular
│ ├── **init**.py
│ ├── main.py # Script orquestador principal
│ ├── module_data.py # Clase DataProcessor (Carga, Limpieza, FE, Split)
│ ├── module_ml.py # Clase ModelEvaluator y Factory de Modelos
│ └── module_path.py # Gestión de rutas relativas
└── README.md # Documentación del proyecto

## 📊 Experimentación

Se compararon los siguientes modelos utilizando métricas de Accuracy y ROC-AUC:

- LogisticRegression
- RandomForest
- GradientBoosting
- AdaBoost
- Bagging
- SVC
- DecisionTree
- MLP
- KNN
- GaussianNB
