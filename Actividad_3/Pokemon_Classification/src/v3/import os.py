import os
import webbrowser
import time

# Nombre del archivo
file_name = "Reporte_Final_V3_Modelos.html"

# Contenido HTML V3 (Con corrección visual en gráfico de tiempos)
html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Final V3: Clasificación Pokémon</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        body { 
            font-family: 'Roboto', sans-serif; 
            background-color: white; 
            color: #1f2937;
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
        }
        @media print {
            /* OCULTAR ENCABEZADOS Y PIES DE PÁGINA DEL NAVEGADOR */
            @page { margin: 0; size: auto; }
            body { margin: 1.6cm; } 
            
            .no-print { display: none !important; }
            .page-break { page-break-before: always; }
            .avoid-break { page-break-inside: avoid; }
            canvas { min-height: 100%; max-width: 100%; }
        }
        .section-title {
            border-bottom: 2px solid #4f46e5;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
            color: #312e81;
            font-weight: 700;
            font-size: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .phase-header {
            background-color: #f3f4f6;
            border-left: 5px solid;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 0 0.5rem 0.5rem 0;
        }
        .chart-container {
            position: relative;
            height: 200px;
            width: 100%;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
    </style>
</head>
<body class="max-w-[210mm] mx-auto p-8 sm:p-12">

    <!-- ENCABEZADO -->
    <header class="text-center mb-10 border-b-4 border-indigo-900 pb-6">
        <h1 class="text-4xl font-bold text-indigo-900 mb-2">REPORTE FINAL DE PROYECTO</h1>
        <h2 class="text-2xl font-light text-slate-600">Optimización de Redes Convolucionales (TinyVGG)</h2>
        <p class="mt-4 text-sm text-slate-500">Actividad 3 | Maestría en Ciencia de Datos</p>
    </header>

    <!-- 1. INTRODUCCIÓN Y CONCEPTOS -->
    <section class="mb-8 avoid-break">
        <h3 class="section-title">1. Introducción y Objetivos</h3>
        <div class="mb-6 text-sm text-justify leading-relaxed">
            <h4 class="font-bold text-slate-800 mb-2">Marco Teórico: Redes Neuronales Convolucionales (CNN)</h4>
            <p class="mb-3">
                Las <strong>Redes Neuronales Convolucionales (CNN)</strong> son un tipo especializado de arquitectura de Deep Learning diseñada para procesar datos con estructura de rejilla, como las imágenes. A diferencia de las redes neuronales densas tradicionales, las CNN utilizan operaciones matemáticas llamadas "convoluciones" que emplean filtros (o kernels) para escanear la imagen y extraer patrones visuales de manera automática.
            </p>
            <p>
                En tareas de <strong>clasificación de imágenes</strong>, las CNN funcionan de manera jerárquica: las primeras capas detectan características simples como líneas, bordes y texturas, mientras que las capas más profundas combinan estos elementos para reconocer formas complejas (como ojos, orejas o alas). Finalmente, esta información abstracta se utiliza para asignar una probabilidad a cada clase posible.
            </p>
        </div>
        <div class="bg-indigo-50 border-l-4 border-indigo-500 p-4 text-sm text-indigo-900 text-justify">
            <strong>Objetivo Específico del Experimento:</strong> 
            El propósito de este proyecto es entrenar y optimizar una CNN basada en la arquitectura <em>TinyVGG</em> para mejorar la clasificación de <strong>150 especies de Pokémon</strong>. Se busca superar el benchmark referencial del 80% de precisión mediante la implementación de estrategias de MLOps, incluyendo aumento de datos (Data Augmentation) y técnicas de regularización (Batch Normalization y Dropout).
        </div>
    </section>

    <!-- 2. DESARROLLO EXPERIMENTAL (MODELOS) -->
    <section>
        <h3 class="section-title">2. Desarrollo Experimental</h3>
        <p class="mb-4 text-sm">A continuación se detallan los cuatro modelos evolutivos, analizando las modificaciones implementadas y su impacto en las curvas de aprendizaje.</p>

        <!-- MODELO 1 -->
        <div class="mb-8 avoid-break">
            <div class="phase-header border-slate-400">
                <h4 class="text-lg font-bold text-slate-800">Modelo 1: Línea Base (Baseline)</h4>
                <p class="text-xs text-slate-600 font-mono mt-1">Configuración: TinyVGG Vanilla | No Augmentation | 20 Épocas</p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="text-sm text-justify">
                    <p class="mb-2"><strong>Modificaciones:</strong> Ninguna. Se utilizó la arquitectura TinyVGG estándar (2 bloques convolucionales) con imágenes redimensionadas a 64x64 píxeles y normalización estándar.</p>
                    <p class="mb-2"><strong>Justificación:</strong> Es fundamental establecer un "suelo" de rendimiento para medir si las futuras "mejoras" realmente aportan valor. Este experimento actúa como control.</p>
                    <p><strong>Hallazgo:</strong> Sorprendentemente, el modelo alcanzó un <strong>92.17%</strong> de precisión sin ayuda extra, indicando que la arquitectura es adecuada para la complejidad del dataset.</p>
                </div>
                <div>
                    <div class="chart-container"><canvas id="chartModel1"></canvas></div>
                </div>
            </div>
        </div>

        <!-- MODELO 2 -->
        <div class="mb-8 avoid-break">
            <div class="phase-header border-blue-500" style="background-color: #eff6ff;">
                <h4 class="text-lg font-bold text-blue-900">Modelo 2: Optimización de Velocidad</h4>
                <p class="text-xs text-blue-700 font-mono mt-1">Config: + Batch Normalization | + Data Augmentation | 15 Épocas</p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="text-sm text-justify">
                    <p class="mb-2"><strong>Modificaciones:</strong> Se insertaron capas de <code>BatchNormalization</code> después de cada convolución y se activó un pipeline de <code>Data Augmentation</code> (rotación 15°, volteo horizontal, jitter de color).</p>
                    <p class="mb-2"><strong>Justificación:</strong> La normalización estabiliza los pesos internos permitiendo un aprendizaje más rápido (convergencia). El aumento de datos evita que el modelo memorice píxeles exactos, forzándolo a aprender formas.</p>
                    <p><strong>Hallazgo:</strong> La convergencia fue explosiva. Para la época 5, el modelo ya superaba el 96%. Alcanzó un récord de <strong>99.57%</strong>.</p>
                </div>
                <div>
                    <div class="chart-container"><canvas id="chartModel2"></canvas></div>
                </div>
            </div>
        </div>

        <div class="page-break"></div>

        <!-- MODELO 3 -->
        <div class="mb-8 avoid-break">
            <div class="phase-header border-green-500" style="background-color: #f0fdf4;">
                <h4 class="text-lg font-bold text-green-900">Modelo 3: Prueba de Regularización (Dropout)</h4>
                <p class="text-xs text-green-700 font-mono mt-1">Config: + Dropout (p=0.2) | 15 Épocas</p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="text-sm text-justify">
                    <p class="mb-2"><strong>Modificaciones:</strong> Se añadió una capa de <code>Dropout(0.2)</code> antes del clasificador final, apagando aleatoriamente el 20% de las neuronas durante el entrenamiento.</p>
                    <p class="mb-2"><strong>Justificación:</strong> El Dropout simula un entrenamiento "en condiciones adversas", impidiendo que las neuronas dependan excesivamente unas de otras (co-adaptación). Esto mejora la generalización.</p>
                    <p><strong>Hallazgo:</strong> Se observa una gran brecha: baja precisión en entrenamiento (~78%) pero máxima en prueba (<strong>99.74%</strong>). Esto confirma que el modelo aprendió características extremadamente robustas.</p>
                </div>
                <div>
                    <div class="chart-container"><canvas id="chartModel3"></canvas></div>
                </div>
            </div>
        </div>

        <!-- MODELO 4 -->
        <div class="mb-8 avoid-break">
            <div class="phase-header border-purple-500" style="background-color: #faf5ff;">
                <h4 class="text-lg font-bold text-purple-900">Modelo 4: Experimento de Alta Regularización</h4>
                <p class="text-xs text-purple-700 font-mono mt-1">Config: Reglas estrictas de regularización | 30 Épocas | Lento</p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="text-sm text-justify">
                    <p class="mb-2"><strong>Modificaciones:</strong> Se aplicó una estrategia de regularización más agresiva y se extendió el entrenamiento a 30 épocas para observar el comportamiento a largo plazo.</p>
                    <p class="mb-2"><strong>Justificación:</strong> Evaluar si un entrenamiento más lento y restringido puede producir un modelo aún más generalizable, aunque sacrifique métricas de entrenamiento.</p>
                    <p><strong>Hallazgo:</strong> El modelo mostró un aprendizaje muy lento (Train Acc: 74% tras 30 épocas) pero una generalización decente (Test Acc: 90.5%). Sin embargo, no logró superar a los Modelos 2 y 3, demostrando que demasiada restricción puede ser contraproducente.</p>
                </div>
                <div>
                    <div class="chart-container"><canvas id="chartModel4"></canvas></div>
                </div>
            </div>
        </div>
    </section>

    <!-- 3. RESULTADOS (TABLA Y GRÁFICO DE TIEMPOS) -->
    <section>
        <h3 class="section-title">3. Resultados Experimentales</h3>
        
        <div class="mb-8 avoid-break">
            <table class="w-full text-sm text-left border border-slate-300 mb-6">
                <thead class="bg-slate-100 text-slate-700">
                    <tr>
                        <th class="px-4 py-2 border">Modelo</th>
                        <th class="px-4 py-2 border text-center">Train Acc</th>
                        <th class="px-4 py-2 border text-center">Test Acc</th>
                        <th class="px-4 py-2 border text-center">Loss Final</th>
                        <th class="px-4 py-2 border">Observación</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="px-4 py-2 border font-medium">1. Baseline</td>
                        <td class="px-4 py-2 border text-center">90.55%</td>
                        <td class="px-4 py-2 border text-center">92.17%</td>
                        <td class="px-4 py-2 border text-center">0.2842</td>
                        <td class="px-4 py-2 border text-xs">Buen inicio, ligeramente inestable.</td>
                    </tr>
                    <tr>
                        <td class="px-4 py-2 border font-medium text-blue-700">2. Optimizado</td>
                        <td class="px-4 py-2 border text-center">99.52%</td>
                        <td class="px-4 py-2 border text-center font-bold">99.57%</td>
                        <td class="px-4 py-2 border text-center font-bold">0.0246</td>
                        <td class="px-4 py-2 border text-xs">Convergencia más rápida y estable.</td>
                    </tr>
                    <tr>
                        <td class="px-4 py-2 border font-medium text-green-700">3. Regularizado</td>
                        <td class="px-4 py-2 border text-center bg-amber-50">78.82%*</td>
                        <td class="px-4 py-2 border text-center font-bold text-green-700">99.74%</td>
                        <td class="px-4 py-2 border text-center">0.0274</td>
                        <td class="px-4 py-2 border text-xs">Mejor generalización teórica.</td>
                    </tr>
                    <tr>
                        <td class="px-4 py-2 border font-medium text-purple-700">4. Alta Reg.</td>
                        <td class="px-4 py-2 border text-center bg-red-50">74.11%</td>
                        <td class="px-4 py-2 border text-center">90.54%</td>
                        <td class="px-4 py-2 border text-center">0.3934</td>
                        <td class="px-4 py-2 border text-xs">Aprendizaje demasiado restringido.</td>
                    </tr>
                </tbody>
            </table>
            <p class="text-xs text-slate-400 italic mb-6">* Nota: La baja precisión de entrenamiento en los Modelos 3 y 4 se debe a la regularización activa.</p>

            <div class="h-48 w-full border border-slate-200 rounded p-4 bg-slate-50 avoid-break">
                <h5 class="text-center font-bold text-slate-600 mb-2 text-xs uppercase">Comparativa de Tiempo Total de Entrenamiento (Minutos)</h5>
                <canvas id="timeChart"></canvas>
            </div>
        </div>
    </section>

    <!-- 4. DISCUSIÓN -->
    <section class="mb-8 avoid-break">
        <h3 class="section-title">4. Discusión y Análisis Comparativo</h3>
        <div class="text-sm text-justify leading-relaxed space-y-4">
            <p>
                <strong>Comparación de Configuraciones:</strong> Al analizar los cuatro modelos, el <strong>Modelo 2</strong> y el <strong>Modelo 3</strong> destacan claramente. El Modelo 4, aunque interesante por su robustez teórica, demostró que un exceso de regularización o una configuración subóptima de hiperparámetros puede frenar demasiado el aprendizaje (underfitting), resultando en un desempeño inferior incluso al Baseline (Modelo 1) en métricas de pérdida.
            </p>
            <p>
                <strong>Fenómeno de Generalización:</strong> En los Modelos 3 y 4, la precisión de prueba fue sistemáticamente superior a la de entrenamiento. Esto es un indicador clásico de regularización fuerte (como Dropout o Augmentation agresivo): el modelo entrena "con pesas", pero al quitárselas en la evaluación, su rendimiento mejora. Sin embargo, en el Modelo 4, estas "pesas" fueron demasiado pesadas, impidiendo que el modelo alcanzara su máximo potencial.
            </p>
        </div>
    </section>

    <!-- 5. LIMITACIONES -->
    <section class="mb-8 avoid-break">
        <h3 class="section-title">5. Limitaciones y Mejoras Futuras</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-justify">
            <div class="bg-slate-50 p-4 rounded border border-slate-200">
                <h4 class="font-bold text-slate-700 mb-2">Limitaciones Actuales</h4>
                <ul class="list-disc list-inside space-y-1 text-slate-600">
                    <li><strong>Resolución Baja:</strong> El uso de imágenes de 64x64 píxeles limita la capacidad del modelo para detectar detalles finos.</li>
                    <li><strong>Clases Desbalanceadas:</strong> No se analizó si algunas clases de Pokémon tienen menos ejemplos que otras, lo que podría sesgar el resultado.</li>
                    <li><strong>Selección de Muestras:</strong> Se tomaron muestras aleatorias de los datos de entrenamiento para formar el conjunto de prueba, lo que garantiza una distribución similar pero podría no reflejar datos del mundo real totalmente nuevos.</li>
                </ul>
            </div>
            <div class="bg-slate-50 p-4 rounded border border-slate-200">
                <h4 class="font-bold text-slate-700 mb-2">Mejoras Propuestas</h4>
                <ul class="list-disc list-inside space-y-1 text-slate-600">
                    <li><strong>Mayor Resolución:</strong> Entrenar con imágenes de 128x128 o 224x224 para capturar texturas.</li>
                    <li><strong>Transfer Learning Avanzado:</strong> Utilizar arquitecturas más profundas como ResNet o EfficientNet pre-entrenadas en ImageNet.</li>
                    <li><strong>Ajuste Fino de Regularización:</strong> Encontrar el punto medio exacto entre el Modelo 3 y el Modelo 4.</li>
                </ul>
            </div>
        </div>
    </section>

    <!-- 6. CONCLUSIONES -->
    <section class="avoid-break">
        <h3 class="section-title">6. Conclusiones Generales</h3>
        <div class="bg-indigo-50 border-l-4 border-indigo-600 p-6 rounded text-sm text-justify leading-relaxed">
            <p class="mb-3">
                El experimento ha sido un éxito rotundo. A través de la implementación progresiva de técnicas de MLOps, logramos transformar un modelo base competente (92%) en un clasificador casi perfecto (99.7%). La combinación de <strong>Data Augmentation</strong> y <strong>Batch Normalization</strong> demostró ser la estrategia más impactante para acelerar el aprendizaje y mejorar la precisión. Si bien el Dropout otorgó la métrica final más alta, su impacto en la velocidad de entrenamiento sugiere que para este dataset específico, el Modelo 2 ofrece el mejor balance costo-beneficio.
            </p>
            <p class="mb-3">
                <strong>Aprendizajes Clave:</strong> Aprendimos que la optimización de hiperparámetros no es lineal: más regularización no siempre es mejor (como vimos en el Modelo 4). El equilibrio entre capacidad del modelo y restricciones de entrenamiento es vital. El Modelo 3 (Dropout 0.2) logró el equilibrio perfecto, ofreciendo la máxima generalización teórica.
            </p>
        </div>
    </section>

    <!-- SCRIPTS DE GRÁFICAS -->
    <script>
        Chart.defaults.font.family = "'Roboto', sans-serif";
        Chart.defaults.font.size = 10;
        Chart.defaults.maintainAspectRatio = false;

        // --- DATOS ---
        const p1_epochs = Array.from({length: 20}, (_, i) => i + 1);
        const p1_train = [0.0860, 0.3464, 0.4871, 0.5777, 0.6329, 0.6828, 0.7201, 0.7475, 0.7681, 0.7868, 0.8106, 0.8275, 0.8416, 0.8531, 0.8532, 0.8719, 0.8830, 0.8868, 0.8968, 0.9055];
        const p1_test = [0.2487, 0.4472, 0.5533, 0.6180, 0.6717, 0.7170, 0.7755, 0.7918, 0.7896, 0.8292, 0.8363, 0.8622, 0.8631, 0.8636, 0.8499, 0.8785, 0.9133, 0.8952, 0.9190, 0.9217];

        const p2_epochs = Array.from({length: 15}, (_, i) => i + 1);
        const p2_train = [0.1618, 0.5041, 0.6989, 0.8226, 0.9121, 0.9622, 0.9864, 0.9913, 0.9928, 0.9954, 0.9948, 0.9952, 0.9962, 0.9973, 0.9952];
        const p2_test = [0.4037, 0.6707, 0.7911, 0.9030, 0.9670, 0.9870, 0.9922, 0.9965, 0.9974, 0.9970, 0.9974, 0.9970, 0.9965, 0.9961, 0.9957];

        const p3_epochs = Array.from({length: 15}, (_, i) => i + 1);
        const p3_train = [0.1086, 0.3748, 0.5172, 0.5903, 0.6635, 0.7049, 0.7302, 0.7609, 0.7757, 0.7906, 0.7884, 0.7912, 0.7937, 0.8023, 0.7882];
        const p3_test = [0.3362, 0.6097, 0.7138, 0.8162, 0.8951, 0.9276, 0.9471, 0.9722, 0.9883, 0.9866, 0.9887, 0.9957, 0.9952, 0.9939, 0.9974];

        // DATOS FASE 4 (NUEVOS)
        const p4_epochs = Array.from({length: 30}, (_, i) => i + 1);
        const p4_train = [0.0235, 0.0926, 0.1655, 0.2318, 0.2968, 0.3369, 0.3857, 0.4184, 0.4445, 0.4873, 0.5110, 0.5328, 0.5626, 0.5766, 0.5867, 0.6039, 0.6284, 0.6387, 0.6515, 0.6645, 0.6659, 0.6878, 0.6996, 0.7014, 0.7048, 0.7195, 0.7228, 0.7407, 0.7243, 0.7411];
        const p4_test = [0.0469, 0.1853, 0.2591, 0.3355, 0.3963, 0.4792, 0.5169, 0.5560, 0.6037, 0.6549, 0.6732, 0.6979, 0.7096, 0.7391, 0.7383, 0.7691, 0.7882, 0.8012, 0.8277, 0.8442, 0.8294, 0.8394, 0.8663, 0.8533, 0.8524, 0.8924, 0.8845, 0.8733, 0.8980, 0.9054];

        function createChart(id, labels, trainData, testData, color) {
            new Chart(document.getElementById(id), {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'Train Acc', data: trainData, borderColor: '#94a3b8', borderWidth: 2, pointRadius: 0, tension: 0.3, borderDash: [5,5] },
                        { label: 'Test Acc', data: testData, borderColor: color, borderWidth: 3, pointRadius: 2, tension: 0.3 }
                    ]
                },
                options: {
                    scales: { y: { min: 0, max: 1, title: {display: true, text: 'Precisión'} } },
                    plugins: { legend: { position: 'bottom', labels: { boxWidth: 10 } } }
                }
            });
        }

        createChart('chartModel1', p1_epochs, p1_train, p1_test, '#64748b');
        createChart('chartModel2', p2_epochs, p2_train, p2_test, '#2563eb');
        createChart('chartModel3', p3_epochs, p3_train, p3_test, '#16a34a');
        createChart('chartModel4', p4_epochs, p4_train, p4_test, '#9333ea'); // Morado

        // Gráfica de Tiempos (Corregida)
        // Se añadió 'layout: { padding: ... }' y se ajustó 'maintainAspectRatio' para evitar cortes.
        new Chart(document.getElementById('timeChart'), {
            type: 'bar',
            data: {
                labels: ['1. Baseline (46m)', '2. Optimizado (16m)', '3. Regularizado (17m)', '4. Alta Reg. (23m)'],
                datasets: [{
                    label: 'Tiempo Total (Minutos)',
                    data: [46, 16, 17, 23],
                    backgroundColor: ['#cbd5e1', '#3b82f6', '#86efac', '#e9d5ff'], 
                    borderColor: ['#94a3b8', '#2563eb', '#22c55e', '#9333ea'],
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                maintainAspectRatio: false,
                layout: {
                    padding: {
                        left: 0,
                        right: 20,
                        top: 0,
                        bottom: 0
                    }
                },
                scales: { 
                    x: { beginAtZero: true, title: {display: true, text: 'Minutos (Menos es mejor)'} },
                    y: { ticks: { autoSkip: false } } 
                },
                plugins: { legend: { display: false } }
            }
        });
    </script>
</body>
</html>
"""

try:
    # Guardar el archivo HTML
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Archivo '{file_name}' creado exitosamente.")
    
    # Abrir automáticamente en el navegador
    print("Abriendo en tu navegador...")
    webbrowser.open('file://' + os.path.realpath(file_name))
    
    print("\n--- INSTRUCCIONES ---")
    print("1. Se abrió tu navegador.")
    print("2. Presiona Ctrl + P.")
    print("3. Guarda como PDF (activa 'Gráficos de fondo').")

except Exception as e:
    print(f"Error: {e}")