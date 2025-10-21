# 🩺 Predicción de Glucosa Postprandial con Machine Learning

## 📋 Descripción General

Este proyecto desarrolla y compara **7 modelos de Machine Learning** diferentes para estimar y clasificar los niveles de **glucosa postprandial** (glucosa después de las comidas) basándose en características clínicas y antropométricas del paciente. 

Todos los modelos siguen la misma metodología y pipeline, variando solo el algoritmo de entrenamiento. Los modelos entrenados son:

1. Regresión Lineal
2. Ridge Regression
3. Lasso Regression
4. Random Forest
5. **Gradient Boosting** (usado como ejemplo en esta documentación)
6. Support Vector Machine (SVM)
7. Red Neuronal (MLP)

El modelo es útil para evaluaciones de diabetes y prediabetes, clasificando automáticamente los resultados en categorías clínicas.

---

## 🎯 Objetivo

- Estimar glucosa postprandial a partir de mediciones de glucosa en ayunas o general
- Clasificar automáticamente los resultados en categorías clínicas: **Normal**, **Prediabetes** o **Diabetes**
- Comparar desempeño entre 7 algoritmos diferentes
- Proporcionar predicciones precisas con métricas de desempeño robustas
- Identificar el modelo óptimo para máxima precisión

---

## 🔬 ¿Qué es Glucosa Postprandial?

La glucosa postprandial es la concentración de glucosa en sangre medida **después de la ingesta de alimentos** (típicamente 2 horas post-comida).

| Tipo de Medición | Momento | Rango Normal |
|-----------------|---------|-------------|
| **Glucosa en Ayunas** | Después de 8 horas sin comer | < 100 mg/dL |
| **Glucosa Postprandial** | 2 horas después de comer | < 140 mg/dL |
| **Glucosa Aleatoria** | Cualquier momento | < 140 mg/dL |

**Importancia clínica**:
- La glucosa postprandial es más alta que en ayunas (~40 mg/dL más)
- Es un predictor importante de diabetes
- Refleja la capacidad del páncreas para regular insulina

---

## 📦 Dependencias Requeridas

### Instalación

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Versiones Recomendadas

```bash
pip install pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=1.0.0 \
            matplotlib>=3.4.0 seaborn>=0.11.0 joblib>=1.0.0
```

---

## 🔧 Librerías Utilizadas

### **Pandas**
```python
import pandas as pd
```

| Función | Descripción |
|---------|-------------|
| `pd.read_csv()` | Carga archivo CSV desde Google Drive |
| `df.dropna()` | Elimina filas con valores faltantes |
| `df.apply()` | Aplica función a cada fila para crear categorías |

**Uso**: Manipulación y limpieza de datos tabulares

---

### **NumPy**
```python
import numpy as np
```

- **Función**: Operaciones numéricas subyacentes
- **Uso**: Cálculos de métricas (RMSE, MAE)
- **Por qué**: Base computacional eficiente para scikit-learn

---

### **Scikit-Learn**

#### `train_test_split`
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

- **Función**: Divide datos en 70% entrenamiento y 30% prueba
- **random_state=42**: Garantiza reproducibilidad
- **Ventaja**: Evalúa el modelo en datos nunca vistos

---

#### `OneHotEncoder`
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore")
```

- **Función**: Convierte variable categórica en numéricas binarias
- **Ejemplo**:
  ```
  Entrada:  ["Normal", "Prediabetes", "Diabetes"]
  Salida:   [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  ```
- **handle_unknown="ignore"**: Maneja categorías no vistas en entrenamiento

---

#### `ColumnTransformer`
```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)
```

- **Función**: Aplica transformaciones diferentes por tipo de columna
- **"passthrough"**: Deja columnas numéricas sin cambios
- **OneHotEncoder**: Transforma solo las categóricas

---

#### `Pipeline`
```python
from sklearn.pipeline import Pipeline

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(...))
])
```

- **Función**: Encadena preprocesamiento y modelo
- **Ventaja**: Evita data leakage (fuga de información entre train/test)
- **Flujo**: `Datos Brutos → Preprocessor → Modelo → Predicción`

---

#### `Métricas de Evaluación`
```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

r2 = r2_score(y_real, y_predicho)
rmse = np.sqrt(mean_squared_error(y_real, y_predicho))
mae = mean_absolute_error(y_real, y_predicho)
```

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **R²** | `1 - (SS_res / SS_tot)` | Proporción de varianza explicada (0-1) |
| **RMSE** | `√(Σ(y_real - y_pred)² / n)` | Error promedio en mg/dL |
| **MAE** | `Σ\|y_real - y_pred\| / n` | Error absoluto medio en mg/dL |

---

### **Joblib**
```python
import joblib

# Guardar modelo
joblib.dump(model, "modelo.joblib")

# Cargar modelo
modelo_cargado = joblib.load("modelo.joblib")
```

- **Función**: Serializa modelos entrenados
- **Ventaja**: Reutilizar sin reentrenamiento

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

### Justificación de Features

- **Datos antropométricos** (edad, peso, talla, IMC): Relacionados con metabolismo y resistencia a insulina
- **Presión arterial** (TAS, TAD): Correlacionada con diabetes tipo 2
- **Categoría de glucosa**: Referencia clínica previa del paciente

### Variable Objetivo (Target)
- **"Resultado"**: Medición de glucosa (en ayunas o general) que se transforma en glucosa postprandial

---

## 🧮 Transformación de Datos

### Estimación de Glucosa Postprandial
```python
Glucosa_Post_Estimada = Resultado + 40  # mg/dL
Glucosa_Post_Estimada = max(Glucosa_Post_Estimada, 70)  # Mínimo clínico
```

**Lógica**:
- La glucosa postprandial es típicamente **40 mg/dL mayor** que en ayunas
- Se asegura un **mínimo de 70 mg/dL** (valor clínicamente significativo)
- Esto simula el aumento esperado después de la ingesta de alimentos

### Limpieza de Datos
```python
df_limpio = df.dropna(subset=features_seleccionadas + [target]).copy()
```
- Elimina filas con valores faltantes
- `copy()` evita modificaciones en el DataFrame original

---

## 🏷️ Clasificación Clínica Postprandial

```python
def clasificar_glucosa_post(valor):
    if valor < 140:
        return "Normal"
    elif 140 <= valor <= 199:
        return "Prediabetes"
    else:
        return "Diabetes"
```

### Criterios de Clasificación (OMS/ADA) - Postprandial

| Clasificación | Rango (mg/dL) | Interpretación |
|---------------|---------------|----------------|
| **Normal** | < 140 | Metabolismo de glucosa saludable |
| **Prediabetes** | 140-199 | Riesgo de desarrollar diabetes |
| **Diabetes** | ≥ 200 | Diabetes mellitus diagnosticada |

**Estándares**:
- Basado en **Prueba de Tolerancia Oral a la Glucosa (PTGO)**
- Criterios de la OMS y Asociación Americana de Diabetes (ADA)
- Medición a los 120 minutos post-comida

---

## 🔄 Estructura Común del Código

Todos los 7 modelos siguen esta estructura idéntica:

```
┌─────────────────────────────────────┐
│  1. CARGAR DATOS (CSV)              │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. CREAR CATEGORÍAS DE GLUCOSA     │
│     (Normal/Prediabetes/Diabetes)   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. ESTIMAR GLUCOSA POSTPRANDIAL    │
│     Resultado + 40 mg/dL            │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. SELECCIONAR FEATURES (7)        │
│     y TARGET (Resultado)            │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  5. LIMPIAR DATOS                   │
│     dropna() en target              │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  6. PREPROCESAR                     │
│     Numéricas: passthrough          │
│     Categóricas: OneHotEncoder      │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  7. DIVIDIR DATOS                   │
│     70% train / 30% test            │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  8. CREAR PIPELINE                  │
│     Preprocessor + Modelo           │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  9. ENTRENAR MODELO                 │
│     model.fit(X_train, y_train)     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  10. EVALUAR                        │
│     Calcular R², RMSE, MAE          │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  11. GUARDAR MODELO                 │
│      joblib.dump()                  │
└─────────────────────────────────────┘
```

---

## 📈 Comparativa de Desempeño

Se entrenaron 7 modelos diferentes para predicción de glucosa postprandial:

| Modelo | R² Test | RMSE Test | MAE Test | Recomendación |
|--------|---------|-----------|----------|--------------|
| **Random Forest** 🥇 | 0.8622 | 12.62 mg/dL | 9.69 mg/dL | ⭐ MEJOR |
| **Gradient Boosting** 🥈 | 0.8240 | 14.27 mg/dL | 10.90 mg/dL | ⭐⭐ Buena |
| **Ridge Regression** 🥉 | 0.8237 | 14.28 mg/dL | 11.26 mg/dL | ⭐⭐ Rápida |
| Lasso Regression | 0.8237 | 14.28 mg/dL | 11.24 mg/dL | ⭐⭐ Rápida |
| Regresión Lineal | 0.8233 | 14.29 mg/dL | 11.28 mg/dL | ⭐ Baseline |
| Support Vector Machine | 0.8114 | 14.77 mg/dL | 11.55 mg/dL | ⭐ Moderada |
| Red Neuronal (MLP) | 0.7956 | 15.37 mg/dL | 11.94 mg/dL | ⭐ Menor |

---

## 🎯 Modelos Implementados

### 1. Regresión Lineal
- **Características**: Simple, interpretable, baseline
- **Ventaja**: Muy rápida
- **Desventaja**: No captura no-linealidades
- **Caso de uso**: Comparación base

### 2. Ridge Regression (L2 Regularization)
- **Características**: Regresión lineal con penalización
- **Ventaja**: Evita sobreajuste, muy rápida
- **Desventaja**: Mantiene todas las variables
- **Caso de uso**: Cuando se necesita estabilidad y velocidad

### 3. Lasso Regression (L1 Regularization)
- **Características**: Regresión lineal con sparsity
- **Ventaja**: Selecciona automáticamente features importantes
- **Desventaja**: Puede eliminar variables útiles
- **Caso de uso**: Selección de variables

### 4. Random Forest ⭐ RECOMENDADO
- **Características**: Ensamble de árboles paralelos
- **Ventaja**: Mejor precisión (R²=0.8622), robusto
- **Desventaja**: Requiere más memoria
- **Caso de uso**: Máxima precisión en producción

### 5. Gradient Boosting
- **Características**: Ensamble secuencial de árboles débiles
- **Ventaja**: Muy buena precisión (R²=0.8240), captura no-linealidades
- **Desventaja**: Lento en entrenamiento
- **Caso de uso**: Datos complejos

### 6. Support Vector Machine (SVM)
- **Características**: Busca hiperplano óptimo
- **Ventaja**: Bueno en espacios de alta dimensión
- **Desventaja**: R² menor (0.8114)
- **Caso de uso**: Cuando hay muchas features

### 7. Red Neuronal (MLP)
- **Características**: Redes con capas densas
- **Ventaja**: Flexible, aprende patrones complejos
- **Desventaja**: Desempeño menor (R²=0.7956), requiere más datos
- **Caso de uso**: Datasets muy grandes

---

## 🔨 Ejemplo de Código: Gradient Boosting

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# 1. Cargar datos
df = pd.read_csv("Glucosa_Unique_Completo.csv")

# 2. Crear categorías
def clasificar_glucosa(valor):
    if valor <= 100:
        return "Normal"
    elif valor <= 125:
        return "Prediabetes"
    else:
        return "Diabetes"

df["Categoria_Glucosa"] = df["Resultado"].apply(clasificar_glucosa)

# 3. Seleccionar features
features_seleccionadas = [
    "Edad_Años", "peso", "talla", "imc", "tas", "tad", "Categoria_Glucosa"
]
target = "Resultado"

df_limpio = df.dropna(subset=features_seleccionadas + [target]).copy()
X = df_limpio[features_seleccionadas]
y = df_limpio[target]

# 4. Preprocesador
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(exclude=["int64", "float64"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# 5. Pipeline con Gradient Boosting
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# 6. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 7. Entrenar
model.fit(X_train, y_train)

# 8. Evaluar
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
rmse_train = (mean_squared_error(y_train, y_pred_train))**0.5
mae_train = mean_absolute_error(y_train, y_pred_train)

r2_test = r2_score(y_test, y_pred_test)
rmse_test = (mean_squared_error(y_test, y_pred_test))**0.5
mae_test = mean_absolute_error(y_test, y_pred_test)

print("="*50)
print("EVALUACIÓN DEL MODELO GRADIENT BOOSTING")
print("="*50)
print(f"[ENTRENAMIENTO] R²={r2_train:.3f} | RMSE={rmse_train:.2f} mg/dL | MAE={mae_train:.2f} mg/dL")
print(f"[PRUEBA       ] R²={r2_test:.3f} | RMSE={rmse_test:.2f} mg/dL | MAE={mae_test:.2f} mg/dL")
print("="*50)

# 9. Guardar
joblib.dump(model, "modelo_gradient_boosting.joblib")
print(f"✅ Modelo guardado")
```

---

## 💾 Cómo Usar un Modelo Entrenado

```python
import joblib
import pandas as pd

# Cargar modelo
modelo = joblib.load("modelo_gradient_boosting.joblib")

# Preparar datos nuevos
X_nuevo = pd.DataFrame({
    "Edad_Años": [45],
    "peso": [75],
    "talla": [170],
    "imc": [26],
    "tas": [120],
    "tad": [80],
    "Categoria_Glucosa": ["Normal"]
})

# Predecir glucosa postprandial
prediccion = modelo.predict(X_nuevo)
print(f"Glucosa Postprandial Estimada: {prediccion[0]:.2f} mg/dL")

# Clasificar resultado
def clasificar(valor):
    if valor < 140:
        return "Normal"
    elif 140 <= valor <= 199:
        return "Prediabetes"
    else:
        return "Diabetes"

clasificacion = clasificar(prediccion[0])
print(f"Clasificación: {clasificacion}")
```

---

## 📊 Interpretación de Métricas

### R² (Coeficiente de Determinación)
- **Rango**: 0 a 1 (más alto es mejor)
- **Interpretación**: Proporción de varianza explicada
- **Ejemplo**: R²=0.86 = El modelo explica el 86% de la variabilidad

### RMSE (Root Mean Squared Error)
- **Unidad**: mg/dL
- **Interpretación**: Error promedio esperado
- **Ejemplo**: RMSE=12.62 mg/dL significa error promedio de ±12.62 mg/dL

### MAE (Mean Absolute Error)
- **Unidad**: mg/dL
- **Interpretación**: Error absoluto promedio
- **Ejemplo**: MAE=9.69 mg/dL significa desviación promedio de 9.69 mg/dL

---

## 🏥 Interpretación Clínica de Errores

**Error de ±12-15 mg/dL (RMSE de Random Forest)**:
- ✅ Aceptable para screening inicial
- ✅ Requiere confirmación con prueba de laboratorio
- ✅ Útil para clasificación (Normal/Prediabetes/Diabetes)

**Por qué es importante**:
- Glucosa Normal: < 140 mg/dL
- Prediabetes: 140-199 mg/dL
- Diabetes: ≥ 200 mg/dL
- **Error de ±12 mg/dL puede cambiar clasificación en fronteras**

**Recomendación clínica**:
- Usar modelo para **screening inicial**
- Confirmar con **prueba de laboratorio**
- **No reemplaza diagnóstico médico profesional**

---

## 📁 Estructura de Archivos

```
proyecto-glucosa-postprandial/
│
├── README.md                                    # Este archivo
├── Glucosa_Unique_Completo.csv                  # Dataset
│
├── modelos/
│   ├── train_linear_regression.py
│   ├── train_ridge_regression.py
│   ├── train_lasso_regression.py
│   ├── train_random_forest.py
│   ├── train_gradient_boosting.py               # Ejemplo del código
│   ├── train_svm.py
│   └── train_mlp.py
│
└── modelos_entrenados/
    ├── modelo_linear_regression.joblib
    ├── modelo_ridge_regression.joblib
    ├── modelo_lasso_regression.joblib
    ├── modelo_random_forest.joblib
    ├── modelo_gradient_boosting.joblib
    ├── modelo_svm.joblib
    └── modelo_mlp.joblib
```

---

## ✅ Ventajas de Esta Estructura

- ✅ **Modular**: Fácil agregar nuevos modelos
- ✅ **Reproducible**: random_state garantiza resultados idénticos
- ✅ **Escalable**: Preprocesamiento automatizado
- ✅ **Reutilizable**: Pipeline encapsulado
- ✅ **Comparable**: Todos los modelos con misma metodología
- ✅ **Clínico**: Clasificación automática según OMS/ADA

---

## ⚠️ Limitaciones

- ⚠️ Requiere datos de calidad bien estructurados
- ⚠️ Error de ±12-15 mg/dL requiere confirmación médica
- ⚠️ No sustituye diagnóstico profesional
- ⚠️ Modelos específicos para predicción de glucosa postprandial

---

## 🚀 Mejoras Futuras

1. Validación cruzada K-Fold para mayor robustez
2. Hiperparámetro tuning automático (Grid Search)
3. Feature importance analysis
4. Comparativa visual con gráficos
5. API REST para despliegue en servidor
6. Dashboard interactivo (Streamlit/Dash)

---

## 📖 Referencias Clínicas

- OMS (2006): Definición y diagnóstico de diabetes mellitus
- ADA Standards of Care: Criterios de clasificación de glucosa postprandial
- PTGO (Prueba de Tolerancia Oral a la Glucosa): Medición a 120 minutos
- Friedman (2001): Gradient Boosting Machines
- Chen & Guestrin (2016): XGBoost - Scalable Tree Boosting

---

## 👤 Autor

[Tu nombre/equipo]

## 📝 Licencia

[Especifica la licencia: MIT, Apache 2.0, etc.]

---

## 📧 Contacto

Para preguntas o sugerencias: [tu email]

---

**Última actualización**: Octubre 2025
**Versión**: 1.0
