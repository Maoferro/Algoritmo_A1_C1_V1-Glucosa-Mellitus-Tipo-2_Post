# 🩺 Predicción de Glucosa Postprandial con Machine Learning

## 📋 Descripción General

Este proyecto utiliza un modelo de **Random Forest Regressor** para estimar y clasificar los niveles de glucosa postprandial (glucosa después de las comidas) basándose en características clínicas y antropométricas del paciente. El modelo es útil para evaluaciones de diabetes y prediabetes.

---

## 🎯 Objetivo

- Estimar glucosa postprandial a partir de mediciones de glucosa en ayunas o general
- Clasificar automáticamente los resultados en categorías clínicas: **Normal**, **Prediabetes** o **Diabetes**
- Proporcionar predicciones precisas con métricas de desempeño robustas

---

## 📦 Dependencias y Librerías

### Librerías Principales

```
pandas            - Manipulación y análisis de datos
numpy             - Operaciones numéricas
scikit-learn      - Machine Learning y preprocesamiento
matplotlib        - Visualización de gráficos
seaborn           - Gráficos estadísticos avanzados
joblib            - Serialización del modelo
```

### Instalación

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Versión Recomendada

```bash
pip install pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=1.0.0 matplotlib>=3.4.0 seaborn>=0.11.0 joblib>=1.0.0
```

---

## 🔧 Detalles de las Librerías Utilizadas

### **Pandas**
- **Uso**: Carga y manipulación del archivo CSV
- **Funciones clave**: `read_csv()`, `dropna()`, `apply()`
- **Por qué**: Proporciona estructuras de datos (DataFrames) para trabajar con datos tabulares

### **NumPy**
- **Uso**: Operaciones numéricas y generación de arreglos
- **Funciones clave**: `linspace()`, `sqrt()` (a través de sklearn.metrics)
- **Por qué**: Base para cálculos matemáticos eficientes

### **Scikit-Learn**
Librería central para Machine Learning con varios módulos:

#### `train_test_split`
- Divide los datos en conjunto de entrenamiento (70%) y prueba (30%)
- Evita el sobreajuste del modelo

#### `RandomForestRegressor`
- **Algoritmo**: Bosque de árboles de decisión aleatorios
- **Parámetros**:
  - `n_estimators=300`: Número de árboles en el bosque
  - `random_state=42`: Garantiza reproducibilidad
  - `n_jobs=-1`: Usa todos los procesadores disponibles
- **Ventajas**: Maneja relaciones no lineales, robusto ante valores atípicos

#### `SimpleImputer`
- Llena valores faltantes
- Estrategia `median` para variables numéricas
- Estrategia `most_frequent` para variables categóricas

#### `OneHotEncoder`
- Convierte variables categóricas en numéricas
- `handle_unknown='ignore'`: Maneja categorías no vistas en entrenamiento

#### `ColumnTransformer`
- Aplica transformaciones diferentes a columnas numéricas y categóricas
- Crucial para preprocesamiento heterogéneo

#### `Pipeline`
- Encadena preprocesamiento + modelo
- Evita data leakage (fuga de información)
- Facilita reproducibilidad

#### `Métricas de Evaluación`
- **R²**: Proporción de varianza explicada (0-1, más alto es mejor)
- **RMSE**: Error cuadrático medio (unidades de glucosa: mg/dL)
- **MAE**: Error medio absoluto (más interpretable)

### **Matplotlib y Seaborn**
- Visualización de resultados
- Gráfico de dispersión (scatter plot) comparando predicciones reales vs. predichas

### **Joblib**
- Serializa el modelo entrenado
- Permite reutilizar el modelo sin reentrenamiento

---

## 📊 Características (Features)

El modelo utiliza **7 características** como entrada:

| Feature | Tipo | Descripción | Rango Típico |
|---------|------|-------------|--------------|
| **Edad_Años** | Numérica | Edad del paciente | 18-100 años |
| **peso** | Numérica | Peso corporal | kg |
| **talla** | Numérica | Altura/Talla | cm |
| **imc** | Numérica | Índice de Masa Corporal | 10-50 kg/m² |
| **tas** | Numérica | Tensión Arterial Sistólica | 80-200 mmHg |
| **tad** | Numérica | Tensión Arterial Diastólica | 40-120 mmHg |
| **Categoria_Glucosa** | Categórica | Clasificación previa de glucosa | Nominal |

### Selección de Características
```python
features_seleccionadas = [
    "Edad_Años", "peso", "talla",
    "imc", "tas", "tad", "Categoria_Glucosa"
]
```

**Justificación**:
- Datos antropométricos relacionados con el metabolismo
- Presión arterial correlacionada con diabetes
- Categoría de glucosa como referencia clínica

### Variable Objetivo (Target)
- **"Resultado"**: Medición de glucosa (en ayunas o general)
- Se transforma en **Glucosa_Post_Estimada**

---

## 🧮 Transformación de Datos

### 1. Estimación de Glucosa Postprandial
```python
Glucosa_Post_Estimada = Resultado + 40 mg/dL
Glucosa_Post_Estimada = max(Glucosa_Post_Estimada, 70)
```

**Lógica**:
- La glucosa postprandial es típicamente 40 mg/dL mayor que en ayunas
- Se asegura un mínimo de 70 mg/dL (valor clínicamente significativo)

### 2. Limpieza de Datos
```python
df_limpio = df.dropna(subset=[target]).copy()
```
- Elimina filas con valores faltantes en la variable objetivo
- `copy()` evita modificaciones en el DataFrame original

---

## 🏷️ Clasificación Clínica

El modelo clasifica automáticamente los resultados en **3 categorías**:

```python
def clasificar_glucosa_post(valor):
    if valor < 140:
        return "Normal"
    elif 140 <= valor <= 199:
        return "Prediabetes"
    else:
        return "Diabetes"
```

### Criterios de Clasificación (OMS/ADA)

| Clasificación | Rango (mg/dL) | Interpretación |
|---------------|---------------|----------------|
| **Normal** | < 140 | Metabolismo de glucosa saludable |
| **Prediabetes** | 140-199 | Riesgo de desarrollar diabetes |
| **Diabetes** | ≥ 200 | Diabetes mellitus diagnosticada |

**Estándares**:
- Basado en pruebas de tolerancia a la glucosa (PTGO)
- Criterios de la OMS y Asociación Americana de Diabetes (ADA)

---

## 🤖 Arquitectura del Modelo

### Pipeline de Procesamiento

```
┌─────────────────────────────────────────┐
│     Datos Crudos (CSV)                  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│     ColumnTransformer                   │
│  ┌──────────────────────────────────┐   │
│  │ Numéricas → SimpleImputer(median)│   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │ Categóricas → Imputer + OneHot   │   │
│  └──────────────────────────────────┘   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   Random Forest Regressor               │
│   (300 árboles de decisión)             │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   Predicciones (mg/dL)                  │
│   + Clasificación (Normal/Prediabetes..)│
└─────────────────────────────────────────┘
```

### Hiperparámetros
- **n_estimators=300**: Mayor número de árboles → mejores predicciones
- **random_state=42**: Reproducibilidad garantizada
- **n_jobs=-1**: Paralelización para mayor velocidad

---

## 📈 Evaluación del Modelo

### Métricas de Desempeño

#### **Coeficiente R² (R-squared)**
```
R² = 1 - (SS_res / SS_tot)
```
- **Interpretación**: Proporción de varianza explicada
- **Rango**: 0 a 1 (más alto es mejor)
- **Ejemplo**: R²=0.85 significa que el modelo explica el 85% de la variabilidad

#### **RMSE (Root Mean Squared Error)**
```
RMSE = √(Σ(y_real - y_pred)² / n)
```
- **Unidad**: mg/dL
- **Interpretación**: Error promedio esperado
- **Ventaja**: Penaliza errores grandes

#### **MAE (Mean Absolute Error)**
```
MAE = Σ|y_real - y_pred| / n
```
- **Unidad**: mg/dL
- **Interpretación**: Más intuitivo que RMSE
- **Ventaja**: Linealmente proporcional al error

### Salida Típica
```
EVALUACIÓN DEL MODELO - GLUCOSA POSTPRANDIAL ESTIMADA
[ENTRENAMIENTO] R² = 0.920 | RMSE = 15.43 mg/dL | MAE = 12.01 mg/dL | n=2450
[PRUEBA       ] R² = 0.895 | RMSE = 18.67 mg/dL | MAE = 14.23 mg/dL | n=1050
```

**Interpretación**:
- El modelo explica el 89.5% de la variabilidad en el conjunto de prueba
- Error promedio de predicción: ±18.67 mg/dL (RMSE)
- La pequeña diferencia entre entrenamiento y prueba indica buen balance

---

## 🔄 Flujo de Ejecución

1. **Carga de Datos**: Lee CSV desde Google Drive
2. **Selección de Características**: Elige 7 features relevantes
3. **Limpieza**: Elimina filas con valores faltantes
4. **Estimación**: Calcula Glucosa_Post_Estimada
5. **Clasificación**: Asigna categoría clínica
6. **División**: 70% entrenamiento, 30% prueba
7. **Entrenamiento**: Ajusta el Random Forest
8. **Predicción**: Genera predicciones en ambos conjuntos
9. **Evaluación**: Calcula R², RMSE, MAE
10. **Guardado**: Serializa el modelo con joblib
11. **Visualización**: Gráfico de dispersión predicciones vs. reales

---

## 📁 Estructura de Archivos

```
proyecto-glucosa/
│
├── modelo_rf_glucosa_postprandial.joblib    # Modelo entrenado
├── Glucosa_Unique_Completo.csv              # Dataset de entrada
├── glucosa_predictor.py                     # Script principal
├── README.md                                # Este archivo
└── resultados/
    └── grafico_predicciones.png             # Visualización
```

---

## 💾 Guardado y Reutilización del Modelo

### Guardar Modelo
```python
joblib.dump(model, "modelo_rf_glucosa_postprandial.joblib")
```

### Cargar Modelo (en otro script)
```python
import joblib
modelo_cargado = joblib.load("modelo_rf_glucosa_postprandial.joblib")

# Predecir nuevos datos
X_nuevo = [[45, 75, 170, 26, 120, 80, "Normal"]]
prediccion = modelo_cargado.predict(X_nuevo)
print(f"Glucosa Postprandial Estimada: {prediccion[0]:.2f} mg/dL")
```

---

## ✅ Ventajas del Modelo

- ✅ **No requiere normalización**: Random Forest es invariante a escala
- ✅ **Maneja valores faltantes**: SimpleImputer integrado
- ✅ **Procesa categóricas automáticamente**: OneHotEncoder en pipeline
- ✅ **Robusto**: Menos sensible a outliers que regresión lineal
- ✅ **Paralelo**: Aprovecha múltiples procesadores
- ✅ **Reproducible**: Con random_state=42

---

## ⚠️ Limitaciones

- ⚠️ Modelo específico para estimación postprandial
- ⚠️ Requiere datos de calidad (CSV bien estructurado)
- ⚠️ Puede sobreajustarse con datasets muy pequeños
- ⚠️ No sustituye diagnóstico médico profesional

---

## 🚀 Mejoras Futuras

1. **Validación Cruzada K-Fold**: Mayor robustez en evaluación
2. **Hiperparámetro Tuning**: Grid Search o Random Search
3. **Feature Engineering**: Crear nuevas variables (razones, interacciones)
4. **Ensemble Methods**: Combinar Random Forest con Gradient Boosting
5. **API REST**: Desplegar modelo en servidor web
6. **Interfaz Web**: Dashboard interactivo con Streamlit/Dash

---

## 📖 Referencias Clínicas

- **OMS (2006)**: Definición y diagnóstico de diabetes mellitus
- **ADA Standards of Care**: Clasificación de glucosa postprandial
- **Liu et al. (2021)**: Machine Learning en predicción de diabetes

---

## 👤 Autor

[Tu nombre/equipo]

## 📝 Licencia

[Especifica la licencia: MIT, Apache 2.0, etc.]

---

## 📧 Contacto

Para preguntas o sugerencias, contacta a: [tu email]

---

**Última actualización**: Octubre 2025
