# Proyecto Final - Predicción de Precios de Vivienda en California

## Descripción General

Este proyecto desarrolla una solución completa de ciencia de datos para la predicción del precio medio de viviendas en distritos de California, abarcando todo el ciclo de vida del modelo: desde la ingesta de datos hasta su despliegue en producción mediante una API.

El enfoque adoptado replica un entorno profesional, integrando buenas prácticas de análisis exploratorio, ingeniería de variables, modelado avanzado y exposición del modelo como servicio.

## Objetivo

Construir un modelo predictivo robusto que estime el valor de viviendas a partir de variables socioeconómicas y geográficas, garantizando:

- Alto desempeño predictivo (bajo RMSE)
- Generalización adecuada (control del overfitting)
- Reproducibilidad del pipeline
- Despliegue funcional en entorno tipo producción


## Enfoque Metodológico

El desarrollo se estructuró en cuatro fases principales:

## 1. Análisis Exploratorio de Datos (EDA)

Se realizó un análisis profundo sobre el conjunto de entrenamiento, identificando:

- Distribuciones sesgadas en variables clave (ej. `median_income`)
- Presencia de outliers relevantes en variables como `total_rooms` y `population`
- Alta correlación entre ingreso medio (`median_income`) y el precio de vivienda
- Importancia de la ubicación geográfica (latitud/longitud)

Se utilizaron visualizaciones como:

- Histogramas
- Mapas geográficos
- Scatter plots
- Matrices de correlación

 Hallazgo clave:  
El ingreso y la ubicación son los principales drivers del precio.

## 2. Limpieza y Feature Engineering

Se diseñó un pipeline de transformación robusto que incluye:

### Tratamiento de datos faltantes
- Imputación de variables numéricas mediante estrategias estadísticas

### Codificación de variables categóricas
- One-Hot Encoding para `ocean_proximity`, evitando asumir orden artificial

### Ingeniería de variables (Feature Engineering)

Se crearon variables con mayor capacidad explicativa:

- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`
- `rooms_per_person`
- `geo_interaction` (latitud × longitud)

Además, se aplicaron transformaciones logarítmicas para reducir sesgo:

- `log_total_rooms`
- `log_total_bedrooms`
- `log_population`
- `log_households`
- `log_median_income`

### Escalado de variables
- Estandarización (`StandardScaler`) para mejorar estabilidad en modelos

Resultado:  Se generó un dataset más representativo y apto para modelado avanzado.

## 3. Modelado y Selección del Modelo

Se evaluaron múltiples algoritmos:

- Linear Regression
- SGD Regressor
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

### Métrica utilizada:
- RMSE (Root Mean Squared Error)

### Validación:
- Cross Validation (k-fold)

### Optimización:
- GridSearchCV y RandomizedSearchCV
- Ajuste de hiperparámetros (depth, learning rate, subsampling, etc.)


## Modelo Final Seleccionado: XGBoost

Después de múltiples iteraciones, el modelo seleccionado fue:

**XGBoost optimizado + selección de variables (Top 18)**

### Resultados:

- RMSE Test: **≈ 44,500**
- Gap controlado entre entrenamiento y prueba
- Mejor capacidad de generalización frente a Random Forest

Interpretación: El modelo logra un balance óptimo entre precisión y estabilidad, evitando sobreajuste excesivo.

## Feature Selection

Se implementó selección de variables basada en importancia del modelo:

- Comparación entre Top 10, 15, 20 y todas las variables
- Mejor desempeño con **Top 18 features**

Insight clave: Más variables no necesariamente mejoran el modelo.

## 4. Despliegue en Producción (FastAPI)

El modelo final fue desplegado mediante una API REST utilizando FastAPI.

### 🔧 Características:

- Carga del modelo serializado (`joblib`)
- Recepción de datos vía JSON
- Aplicación del pipeline completo en tiempo real:
  - Imputación
  - Feature engineering
  - Escalado
  - Codificación
- Predicción en línea

### Endpoint:

http
POST /predict 
#### Input 
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"

#### Output 
predicted_median_house_value: 440264.38

### Conclusiones de Negocio
El ingreso medio es el principal predictor del valor inmobiliario
La proximidad al océano incrementa significativamente los precios
Las variables derivadas capturan mejor la densidad habitacional
Modelos avanzados (XGBoost) superan ampliamente a modelos lineales

Aplicación real: Este modelo puede ser utilizado para:

Tasación automatizada de viviendas
Evaluación de inversiones inmobiliarias
Análisis de mercado

### Conclusión Final

El proyecto demuestra la construcción de una solución end-to-end de ciencia de datos, integrando análisis, modelado y despliegue.

Se logró no solo un modelo predictivo robusto, sino también su operacionalización en un entorno productivo, cumpliendo con estándares reales de la industria.


