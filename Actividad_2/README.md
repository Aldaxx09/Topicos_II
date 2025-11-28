# 🔬 Tópicos II - Actividad 2: Pipeline Modular con MLflow

Este proyecto implementa un flujo de trabajo (pipeline) de Machine Learning modular para la clasificación de diagnósticos médicos (Dataset WiDS Datathon 2024).

El sistema automatiza el preprocesamiento, la selección de características y la evaluación comparativa de múltiples modelos, registrando todos los experimentos mediante **MLflow**.

## 🚀 Características Principales

* **Arquitectura Modular:** Código organizado en módulos: procesamiento (`module_data.py`), modelado (`module_ml.py`) y orquestación (`main.py`).
* **MLflow Tracking:** Registro automático de métricas (ROC-AUC, Accuracy), parámetros y modelos.
* **Consola Limpia:** Ejecución optimizada para mostrar solo la información relevante (sin advertencias ni barras de carga).
* **Organización:** Los experimentos se guardan automáticamente en la carpeta `mlruns` fuera del código fuente.

## 📂 Estructura del Proyecto

```text
├── data/
│   └── training.csv         # Dataset de entrada
├── mlruns/                  # Historial de experimentos (Se crea automáticamente al ejecutar)
├── src/                     # Código Fuente
│   ├── main.py              # Script Principal (Orquestador)
│   ├── module_data.py       # Procesamiento de Datos
│   └── module_ml.py         # Entrenamiento y Evaluación
└── README.md                # Este archivo

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

⚙️ Requisitos
Asegúrate de tener instalado Python 3.9+. Instala las librerías necesarias ejecutando:
Bash
pip install pandas numpy scikit-learn mlflow

Entendido. Aquí tienes el contenido completo para tu archivo README.md.

Aunque es un archivo de documentación (Markdown), te lo pongo en un bloque de código para que puedas copiarlo y pegarlo fácilmente en tu editor.

Instrucciones: Crea un archivo llamado README.md en la carpeta raíz de tu proyecto (al mismo nivel que la carpeta src) y pega el siguiente contenido:

Markdown

# 🔬 Tópicos II - Actividad 2: Pipeline Modular con MLflow

Este proyecto implementa un flujo de trabajo (pipeline) de Machine Learning modular para la clasificación de diagnósticos médicos (Dataset WiDS Datathon 2024).

El sistema automatiza el preprocesamiento, la selección de características y la evaluación comparativa de múltiples modelos, registrando todos los experimentos mediante **MLflow**.

## 🚀 Características Principales

* **Arquitectura Modular:** Código organizado en módulos: procesamiento (`module_data.py`), modelado (`module_ml.py`) y orquestación (`main.py`).
* **MLflow Tracking:** Registro automático de métricas (ROC-AUC, Accuracy), parámetros y modelos.
* **Consola Limpia:** Ejecución optimizada para mostrar solo la información relevante (sin advertencias ni barras de carga).
* **Organización:** Los experimentos se guardan automáticamente en la carpeta `mlruns` fuera del código fuente.

## 📂 Estructura del Proyecto

├── data/
│   └── training.csv         # Dataset de entrada
├── mlruns/                  # Historial de experimentos (Se crea automáticamente al ejecutar)
├── src/                     # Código Fuente
│   ├── main.py              # Script Principal (Orquestador)
│   ├── module_data.py       # Procesamiento de Datos
│   └── module_ml.py         # Entrenamiento y Evaluación
└── README.md                # Este archivo
⚙️ Requisitos
Asegúrate de tener instalado Python 3.9+. Instala las librerías necesarias ejecutando:
Bash
pip install pandas numpy scikit-learn mlflow


▶️ Guía de Ejecución
1. Entrenar el Pipeline
Para ejecutar la experimentación completa, navega a la carpeta src y ejecuta el archivo main.py:
Bash
cd src
python main.py
