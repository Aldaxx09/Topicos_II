readme_content = """

# 🐉 Actividad 3: Clasificación de Pokémon con CNN (TinyVGG)

Este repositorio contiene la implementación de un pipeline de **Deep Learning** modular diseñado para clasificar imágenes de **150 especies de Pokémon**. El proyecto fue desarrollado como parte de la asignatura _Tópicos Selectos de Grandes Bases de Datos_ de la **Maestría en Ciencia de Datos**.

## 🎯 Objetivo del Proyecto

El objetivo principal fue entrenar y optimizar una **Red Neuronal Convolucional (CNN)** para superar el benchmark referencial de **80% de precisión** en la clasificación de Pokémon.

El enfoque se centró en:

1. Migrar de notebooks monolíticos a **scripts modulares (MLOps)**.

2. Implementar la arquitectura **TinyVGG** desde cero.

3. Aplicar técnicas de optimización (**Data Augmentation**, **Batch Normalization**, **Dropout**) para maximizar la generalización.

## 📂 Estructura del Repositorio

El código sigue el principio de **Separación de Responsabilidades (SoR)**:

```
Pokemon_Classification/
├── data/                   # Directorio de datos (no incluido en git)
│   ├── train/              # Imágenes de entrenamiento
│   └── test/               # Imágenes de prueba
├── models/                 # Modelos entrenados (.pth)
├── notebooks/               # ipynb
├── src/                    # Código fuente
│   ├── data_setup.py       # Carga y preprocesamiento (DataLoaders + Augmentation)
│   ├── model_builder.py    # Definición de arquitecturas (TinyVGG, ImprovedTinyVGG)
│   ├── engine.py           # Bucle de entrenamiento y evaluación
│   ├── utils.py            # Funciones auxiliares (guardado de modelos)
│   ├── get_data.py         # Script para dividir dataset (Train/Test split)
│   └── train.py            # Orquestador principal (Main script)
├── reportes/               # Reportes generados (HTML/PDF)
└── README.md               # Documentación del proyecto

```

## 🧪 Metodología Experimental

Se llevó a cabo un proceso iterativo de 4 fases para aislar el impacto de cada técnica de optimización:

| Modelo              | Configuración        | Descripción                                                               |
| ------------------- | -------------------- | ------------------------------------------------------------------------- |
| **1. Baseline**     | `TinyVGG Vanilla`    | Modelo base sin aumento de datos ni normalización.                        |
| **2. Optimizado**   | `+ Batch Norm`       | Adición de normalización por lotes y aumento de datos (rotación, jitter). |
| **3. Regularizado** | `+ Dropout (0.2)`    | Inclusión de Dropout para forzar robustez y generalización.               |
| **4. Alta Reg.**    | `+ Dropout Agresivo` | Prueba de estrés con regularización estricta y entrenamiento prolongado.  |

## 📊 Resultados Obtenidos

El modelo final superó ampliamente las expectativas, logrando una precisión casi perfecta en el conjunto de prueba.

| Experimento      | Train Acc | Test Acc   | Test Loss  | Observación                               |
| ---------------- | --------- | ---------- | ---------- | ----------------------------------------- |
| **Baseline**     | 90.55%    | 92.17%     | 0.2842     | Buen inicio, convergencia lenta.          |
| **Optimizado**   | 99.52%    | 99.57%     | **0.0246** | **Mejor eficiencia (Tiempo/Acc).**        |
| **Regularizado** | 78.82%    | **99.74%** | 0.0274     | **Mayor robustez teórica.**               |
| **Alta Reg.**    | 74.11%    | 90.54%     | 0.3934     | Underfitting por exceso de restricciones. |

> **Nota:** En el modelo "Regularizado", la baja precisión de entrenamiento (78%) frente a la alta precisión de prueba (99%) es un efecto esperado del **Dropout**, que "apaga" neuronas durante el entrenamiento para evitar la memorización.

### Visualización de Desempeño

_(Ver carpeta `reportes/` para gráficos detallados de curvas de aprendizaje y matrices de confusión)._

## 🚀 Instalación y Uso

1. **Clonar el repositorio:**

   ```
   git clone [https://github.com/Aldaxx09/Topicos_II.git](https://github.com/Aldaxx09/Topicos_II.git)
   cd pokemon-classification

   ```

2. **Instalar dependencias:**

   ```
   pip install torch torchvision matplotlib tqdm

   ```

3. **Preparar los datos:**
   Asegúrate de tener las imágenes en `data/train`. Ejecuta el script para crear el set de prueba:

   ```
   cd src
   python get_data.py

   ```

4. **Entrenar el modelo:**
   Ejecuta el orquestador principal. Puedes ajustar hiperparámetros dentro de `train.py`.

   ```
   python train.py

   ```

## 🧠 Conceptos Clave

- **CNN (Red Neuronal Convolucional):** Arquitectura especializada en procesar datos con estructura de rejilla (imágenes) mediante el uso de filtros que extraen características jerárquicas.

- **Transfer Learning:** Aunque aquí entrenamos desde cero (`TinyVGG`), la estructura modular permite integrar fácilmente modelos pre-entrenados como `EfficientNet`.

- **Data Augmentation:** Técnica utilizada para aumentar artificialmente la diversidad del set de datos mediante transformaciones aleatorias (rotación, espejo), reduciendo el overfitting.

## ✒️ Autor

**Jesus Adahir Copado Crespo** Maestría en Ciencia de Datos - CUCEA
"""

print(readme_content)
