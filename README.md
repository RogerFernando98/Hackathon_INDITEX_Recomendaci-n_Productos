# INDITEX – Sistema de Recomendación de Productos

Sistema completo de recomendación de productos para e‑commerce, basado en datos reales de usuarios, sesiones y productos de Inditex, que combina **preprocesamiento avanzado**, **ingeniería de features** y **modelos de recomendación** para sugerir los **5 productos más relevantes por sesión**.

## Contexto

Este proyecto desarrolla un sistema de recomendación que, a partir de los datos de navegación, usuarios y productos de una plataforma de moda online, es capaz de predecir qué productos tienen mayor probabilidad de ser añadidos al carrito por cada usuario en cada sesión.

## ¿Qué problema soluciona?

En e‑commerce de moda, mostrar productos relevantes es crítico:

- Los catálogos son enormes  
- Los gustos cambian rápido  
- Los usuarios abandonan si no encuentran lo que buscan  

Sin un buen sistema de recomendación:
- Aumenta la tasa de rebote  
- Se pierde conversión  
- Se reduce el valor del cliente  

Este proyecto aborda ese problema usando **machine learning y datos de comportamiento real** para predecir qué productos tienen más probabilidad de ser comprados.

## Contexto del problema

Inditex gestiona millones de interacciones diarias entre usuarios y productos. Cada sesión contiene:

- Navegación por páginas  
- Interacciones con productos  
- Añadidos al carrito  
- Información del usuario (cuando existe)

El reto es **transformar este rastro digital en señales útiles** para recomendar productos relevantes en tiempo real.  
El dataset refleja un escenario realista de e‑commerce moderno, donde existen:
- Usuarios conocidos  
- Usuarios anónimos  
- Usuarios nuevos  
- Sesiones parcialmente observadas  

## Objetivo del proyecto

El objetivo es construir un **recommender system realista y escalable** que permita:

- Analizar el comportamiento de los usuarios en la plataforma  
- Modelar la interacción usuario–producto  
- Generar recomendaciones personalizadas por sesión  
- Evaluar su rendimiento mediante métricas de ranking (NDCG)

## Datos utilizados

El proyecto utiliza cuatro fuentes principales:

### 1️⃣ Users (`users.csv`)
Información agregada por usuario:
- `user_id`
- `country`
- `R` (Recency)
- `F` (Frequency)
- `M` (Monetary)

### 2️⃣ Train (`train.csv`)
Interacciones históricas:

| Variable | Descripción |
|--------|-------------|
| session_id | Identificador de sesión |
| date | Fecha |
| timestamp_local | Timestamp |
| user_id | Usuario (NaN si anónimo) |
| country | País |
| partnumber | Producto |
| device_type | Tipo de dispositivo |
| pagetype | Tipo de página |
| add_to_cart | 1 si se añadió al carrito |


### 3️⃣ Test (`test.csv`)
Igual que Train, pero sin `add_to_cart`.  
Es el dataset donde se generan las recomendaciones.

### 4️⃣ Products (`products.pkl`)
Información de producto:
- `partnumber`
- `discount`
- `cod_section`
- `family`
- `embedding` (vector visual del producto)

### Preprocesamiento aplicado

En `prepare_data.py` se generaron:

- Limpieza de nulos  
- Conversión de fechas  
- One-Hot Encoding  
- Clustering de embeddings  
- Features de popularidad  
- Features de sesión  

Los resultados finales están en:

```bash
data/processed/
├── users_prepared_.parquet
├── train_prepared_.parquet
├── test_prepared_.parquet
├── products_prepared_.parquet
├── session_features_.npy
├── popularity_features_.npy
├── product_embeddings_*.npy
├── test.csv
├── train.csv
└── users.csv
```
## Estructura del proyecto
```bash
HACKATHON_INDITEX_RECOMENDACION_PRODUCTOS/
│
├── data/
│   ├── raw/     # Sin API y sin datos iniciales.
│   └── processed/
├── models/
├── predictions/
├── src/
├── tests/
├── requirements.txt
├── Enunciado.txt
├── .gitignore
└── README.md
```

## Metodología

### 1️⃣ Data Preparation
- Limpieza de datos  
- Encoding de variables  
- Imputación de nulos  
- Clustering de embeddings  
- Ingeniería de features  

### 2️⃣ Task 1 – Queries analíticas  
Resueltas y exportadas a `predictions_1.json`.

### 3️⃣ Task 2 – Métricas de sesión  
Función validada con `pytest`.

### 4️⃣ Task 3 – Recomendador  
Sistema híbrido entrenado y serializado.

## Tecnologías y librerías

Proyecto desarrollado en **Python 3.10**.

- pandas  
- numpy  
- scikit-learn  
- scipy  
- joblib  
- pyarrow  
- pytest  

Instalación:

```bash
pip install -r requirements.txt
```
## Resultados
- Task 1 → JSON validado
- Task 2 → 100% tests pasados
- Task 3 → Recomendaciones generadas
- Modelo serializado


## Próximos pasos
- Ajuste de hiperparámetros
- Modelos secuenciales
- Mejor Cold-Start
- Despliegue como API

## 👤 Autor

Proyecto desarrollado por Roger Fernando Arroyo Herrera.
LinkedIn: [Contáctame por LinkedIn](www.linkedin.com/in/f-arroyo-herrera)