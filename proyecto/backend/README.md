# Mantenimiento preventivo con lógica difusa y algoritmo genético

Aplicación web académica de **mantenimiento predictivo** para una máquina
industrial de manufactura con elementos rotativos. Un usuario introduce las
condiciones actuales de operación y recibe:

- un **índice de riesgo de fallo (0–100)**;
- un **nivel de riesgo** (BAJO / MODERADO / ALTO / CRÍTICO);
- una **acción recomendada de mantenimiento**;
- una explicación sencilla con los **factores** que contribuyeron al resultado.

La parte inteligente combina tres técnicas de IA:

1. **Lógica difusa (Mamdani):** corazón del sistema de inferencia.
2. **PRISM:** mina desde el dataset las reglas difusas que el sistema utiliza,
   cada una con evidencia medible (confianza, cobertura, lift).
3. **Algoritmo genético:** optimiza los parámetros de las funciones de
   pertenencia del sistema difuso frente al dataset.

> **Aviso importante:** el proyecto es académico. El dataset AI4I 2020 es
> sintético y el índice de riesgo es la salida de un sistema difuso: no
> representa una probabilidad estadística exacta ni constituye un sistema
> industrial certificado.

---

## 1. Problema

Predecir cuándo conviene dar mantenimiento a una máquina industrial es una
tarea con incertidumbre. Las condiciones de operación (temperaturas, velocidad
de rotación, torque, desgaste) son continuas y sus límites "normales" vs
"peligrosos" no son nítidos: un valor puede pertenecer parcialmente a más de
una categoría.

## 2. Objetivo

Construir un sistema que transforme cinco variables de operación en un índice
de riesgo continuo y una acción recomendada, justificando el resultado con las
reglas difusas efectivamente activadas.

## 3. Dataset

**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository),
sintético, con 10 000 observaciones. Variable objetivo: `Machine failure`
(339 positivos → dataset desbalanceado).

Referencia oficial:

> Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive
> Maintenance Applications*. AI4I 2020 Predictive Maintenance Dataset.
> UCI Machine Learning Repository. https://doi.org/10.24432/C5H30F

- Sitio: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- El dataset se encuentra en `backend/data/ai4i2020.csv` y **no se vuelve a
  descargar** durante la ejecución.

## 4. Variables de entrada (formulario)

| Variable | Unidad | Rango en datos |
| -------- | ------ | -------------- |
| Air Temperature | K | 295.3 – 304.5 |
| Process Temperature | K | 305.7 – 313.8 |
| Rotational Speed | rpm | 1168 – 2886 |
| Torque | Nm | 3.8 – 76.6 |
| Tool Wear | min | 0 – 253 |

## 5. Variable derivada

Se calcula internamente `thermal_gap = process_temperature − air_temperature`
(brecha térmica, indicadora de estrés térmico). No se solicita al usuario; se
muestra como información calculada en los detalles de la inferencia.

## 6. Arquitectura

```
Usuario → Formulario → API (validación) → thermal_gap → Fuzzificación
       → Reglas difusas → Agregación → Defuzzificación (centroide)
       → Índice de riesgo → Nivel → Acción → Respuesta
```

El algoritmo genético **no corre en cada consulta**: se ejecuta una vez
(`scripts/optimize.py`) y guarda sus resultados en
`artifacts/optimized_parameters.json`, que la aplicación carga al arrancar.

```
Dataset → GA → parámetros óptimos → optimized_parameters.json → Sistema difuso
```

### Backend

```
backend/
├── data/ai4i2020.csv               Dataset (no se re-descarga)
├── artifacts/
│   ├── optimized_parameters.json   Parámetros optimizados por el GA
│   └── mined_rules.json            Base de reglas generada por PRISM
├── main.py                    API FastAPI (arranque, deliberadamente pequeño)
├── config.py                  Constantes centrales: rangos, niveles, acciones,
│                              cargador de reglas, salida, hiperparámetros del GA
├── fuzzy.py                   Sistema difuso Mamdani con scikit-fuzzy
│                              (membresías, reglas, agregación, centroide)
├── prism.py                   Minería de reglas PRISM (discretización,
│                              inducción de reglas, confianza/cobertura/lift)
├── genetic.py                 Algoritmo genético con DEAP (individuo,
│                              selección, crossover, mutación, elitismo)
├── model.py                   Carga parámetros, construye el sistema y
│                              transforma el riesgo en nivel/acción/factores
├── schemas.py                 Modelos request/response de la API (Pydantic)
├── scripts/mine_rules.py      Minería de reglas (CSV → PRISM → JSON → reporte)
├── scripts/optimize.py        Entrenamiento completo (CSV → split → GA → JSON)
├── tests/test_fuzzy.py        Pruebas básicas
├── requirements.txt
└── README.md
```

### Frontend

```
frontend/
└── src/pages/index.astro      Página única: formulario + resultado + detalles
```

## 7. Lógica difusa (Mamdani)

El sistema de inferencia se construye con **scikit-fuzzy** (`skfuzzy.control`):
`Antecedent`, `Consequent`, `Rule`, `ControlSystem` y
`ControlSystemSimulation`. En `fuzzy.py`, la subclase `FuzzySimulation` ordena
las reglas una sola vez y las reutiliza en cada evaluación por lotes para
evitar reconstruir el orden con networkx en cada fila (hasta ~8× más rápido).

1. **Fuzzificación**: cada variable de entrada se evalúa con tres conjuntos
   difusos `LOW / MEDIUM / HIGH` y devuelve grados de pertenencia en `[0, 1]`.
2. **Reglas**: base generada por **PRISM** a partir del dataset (artefacto
   `mined_rules.json`), cargada en `config.py` (`FUZZY_RULES`). Cada regla
   registra confianza, cobertura y lift.
3. **Agregación**: para un antecedente AND se usa `min(...)`; la combinación de
   reglas con el mismo consecuente usa `max(...)` (comportamiento por defecto de
   `skfuzzy.control.Rule`).
4. **Defuzzificación**: **centroide** sobre el dominio de salida `0..100`.

### Variables lingüísticas

| Variable difusa | Conjuntos | Tipo |
| --------------- | --------- | ---- |
| Process Temperature | LOW / MEDIUM / HIGH | trapecio / triángulo / trapecio |
| Thermal Gap | LOW / MEDIUM / HIGH | trapecio / triángulo / trapecio |
| Rotational Speed | LOW / MEDIUM / HIGH | trapecio / triángulo / trapecio |
| Torque | LOW / MEDIUM / HIGH | trapecio / triángulo / trapecio |
| Tool Wear | LOW / MEDIUM / HIGH | trapecio / triángulo / trapecio |

Cada variable se representa con **tres puntos de corte** `p1 < p2 < p3`:

```text
LOW    = [min, min, p1, p2]   (trapezoidal)
MEDIUM = [p1, p2, p3]         (triangular)
HIGH   = [p2, p3, max, max]   (trapezoidal)
```

La salida `risk` tiene cuatro conjuntos: LOW, MEDIUM, HIGH, CRITICAL.

### Reglas difusas generadas por PRISM

Las reglas se **minan desde el dataset** con el algoritmo **PRISM**
(Cendrowska, 1987) en `prism.py` + `scripts/mine_rules.py`. El proceso es
determinista y reproducible:

1. **Discretización:** cada fila del AI4I se convierte en el término de máxima
   pertenencia por variable (`LOW / MEDIUM / HIGH`) usando la partición canónica
   (parámetros por defecto).
2. **Inducción:** PRISM busca condiciones que maximicen la proporción de la
   clase objetivo sobre la porción de datos restante, cubriendo filas hasta
   agotar la clase. Se minan dos clases: **fallo** (confianza ≥ 0.40, cobertura
   ≥ 5) y **sin fallo** (confianza ≥ 0.965, cobertura ≥ 50).
3. **Métricas por regla:** confianza (fracción de filas que cumplen la
   condición y pertenecen a la clase), cobertura (filas que cumplen la
   condición) y lift (cuántas veces más probable es la clase respecto a la
   línea base global, 3.39 % de fallos).

Reglas de fallo descubiertas (confianza ≥ 0.40):

```text
PR1: IF torque IS HIGH AND thermal_gap IS MEDIUM AND tool_wear IS LOW
     AND process_temperature IS HIGH AND rotational_speed IS LOW
                                        THEN risk IS HIGH   (conf 55.6 %, lift 16.4)
PR2: IF torque IS HIGH AND thermal_gap IS MEDIUM AND tool_wear IS LOW
     AND process_temperature IS MEDIUM AND rotational_speed IS LOW
                                        THEN risk IS HIGH   (conf 54.2 %, lift 16.0)
...
```

El patrón dominante que descubre el algoritmo es **torque alto + velocidad de
rotación baja** (sobrecarga), con variantes por desgaste y temperatura. Las
reglas seguras (sin fallo) tienen confianza 100 % y amplia cobertura (p. ej.
`torque IS LOW AND rotational_speed IS LOW AND tool_wear IS LOW` → 1670 filas
sin ningún fallo) y disparan el consecuente LOW. La base completa (7 de fallo +
4 seguras) queda en `artifacts/mined_rules.json` y se imprime en español al
ejecutar la minería.

> El GA **no** genera las reglas: optimiza las funciones de pertenencia. Las
> reglas provienen del algoritmo de minería (PRISM) y son fijas entre corridas.

### Explicación del resultado

No se usa texto fijo: los **factores** se derivan de los antecedentes de las
reglas realmente activadas. Por ejemplo, si se activa una regla con
`tool_wear = HIGH`, el sistema reporta "Desgaste de herramienta elevado".

## 8. Índice de riesgo y niveles

La salida difusa es un número 0–100 llamado **Índice de riesgo** (no
"probabilidad de fallo"). Los límites están centralizados en `config.py`
(`RISK_LEVELS`):

| Rango | Nivel | Acción recomendada |
| ----- | ----- | ------------------ |
| 0–24 | BAJO | Continuar operación |
| 25–49 | MODERADO | Monitorear |
| 50–74 | ALTO | Programar mantenimiento |
| 75–100 | CRÍTICO | Detener y revisar |

## 9. Algoritmo genético

Implementado con **DEAP** (`deap.base`, `deap.creator`, `deap.tools`) en
`genetic.py`. Optimiza los **15 genes** (`p1, p2, p3` de las cinco variables
lingüísticas) para maximizar la **Balanced Accuracy** del sistema difuso frente
al dataset de entrenamiento.

Flujo:

1. Población inicial aleatoria (`tools.initRepeat`), incluyendo los parámetros
   por defecto como individuo de la población.
2. Evaluar fitness (balanced accuracy sobre una muestra estratificada).
3. Selección por torneo (`tools.selTournament`, `k = 3`).
4. Crossover de un punto (`tools.cxOnePoint`, probabilidad 0.8).
5. Mutación gaussiana por gen (`tools.mutGaussian`, probabilidad 0.1).
6. Elitismo (`tools.selBest`, conserva los 2 mejores).
7. Repetir 50 generaciones.
8. Devolver el mejor individuo (`tools.HallOfFame`).

Al decodificar, los tres genes de cada variable se ordenan y se aplica una
**separación mínima** para evitar funciones degeneradas (`p1 < p2 < p3`). Como
`skfuzzy.trapmf` valida `a ≤ b ≤ c ≤ d`, los genes se recortan a `[0, 1]`
antes de decodificar.

Configuración (configurable en `config.py`):

```text
population_size = 30
generations     = 50
elite_count     = 2
crossover_rate  = 0.8
mutation_rate   = 0.1
seed            = 42 (fijo, evolución reproducible)
fitness_sample_size = 600
```

### Fitness

El fitness principal es **Balanced Accuracy** (adecuado por el desbalance de
clases: solo 339 fallos). Para que la evaluación sea viable, el fitness se
calcula sobre una **muestra estratificada de 600 filas** (todos los positivos +
negativos al azar, seed fija): las métricas finales se reportan siempre sobre
el conjunto completo. El modo escalar de skfuzzy es más lento pero estable en
filas sin reglas activas, por lo que la evaluación del fitness recorre cada
fila individualmente. Además se registran Accuracy, Precision, Recall, F1 y la
matriz de confusión. La conversión riesgo → fallo usa `risk >= 50`
(`RISK_THRESHOLD` en config).

> El GA **no reemplaza** la lógica difusa ni calcula el riesgo de cada usuario:
> solo encuentra los parámetros de pertenencia que mejoran el comportamiento
> del sistema difuso frente a los datos.

## 10. División del dataset

70 % entrenamiento / 15 % validación / 15 % test final. Se usó una división
**estratificada y reproducible** (`random_state=42`): aunque el dataset está
ordenado por un identificador de fila (UDI), el orden no representa una
secuencia temporal real que aporte información, y la estratificación garantiza
la misma proporción de fallos en cada partición. El test permanece aislado
hasta el final de la optimización.

## 11. Cómo ejecutar

Requisitos: Python 3.10+ (probado con 3.14), Node 22+ y pnpm.

### Backend

```bash
cd backend
python -m venv .venv                       # opcional
python -m pip install -r requirements.txt  # o usar el venv del workspace
uvicorn main:app --host localhost --port 8001
```

Endpoints:

- `GET  /api/health` → `{"status": "ok"}`
- `POST /api/predict` → índice, nivel, acción, factores, reglas y membresías

Ejemplo de request:

```json
{
  "air_temperature": 300.0,
  "process_temperature": 310.0,
  "rotational_speed": 1500,
  "torque": 40.0,
  "tool_wear": 100
}
```

### Frontend

```bash
cd frontend
pnpm install
pnpm dev          # http://localhost:4321
```

El frontend llama al backend en `http://localhost:8001/api/predict`.

## 12. Cómo ejecutar la minería y la optimización

```bash
cd backend
python -m scripts.mine_rules     # genera artifacts/mined_rules.json + reporte en español
python -m scripts.optimize       # genera artifacts/optimized_parameters.json
```

`scripts/mine_rules.py` imprime cada regla con su confianza, cobertura y lift y
guarda la base difusa en `artifacts/mined_rules.json` (versionado, fuente de
verdad de las reglas). `scripts/optimize.py` genera el JSON con versión, seed,
fecha, hiperparámetros, parámetros óptimos, fitness y métricas antes/después.

Si falta cualquiera de los dos artefactos, la API no ejecuta nada en silencio:
responde con un error claro indicando qué proceso debe ejecutarse primero.

## 13. Resultados de optimización

Comparación experimental sobre el **mismo conjunto de test** (1501 filas,
aislado durante la optimización):

| Métrica | Sin optimizar | Con GA (optimizado) | Δ |
| ------- | ------------- | ------------------- | - |
| Balanced Accuracy | 0.769 | 0.886 | +0.117 |
| Precision | 0.139 | 0.251 | +0.112 |
| Recall | 0.686 | 0.863 | +0.176 |
| F1 | 0.232 | 0.389 | +0.158 |

En validación el sistema optimizado alcanza una Balanced Accuracy de 0.858 y
en entrenamiento de 0.865; el mejor fitness durante la evolución fue 0.880
(sobre la muestra estratificada de fitness). El GA redujo los falsos positivos
en test (de 216 a 131) y subió la precisión del 13.9 % al 25.1 %: una de cada
cuatro alarmas corresponde a un fallo real.

Frente al sistema anterior (reglas manuales R1–R15 con GA), la base generada
por PRISM mejora la **Precision** (0.195 → 0.251) y el **F1** (0.319 → 0.389)
con una Balanced Accuracy similar (0.877 → 0.886) y recall prácticamente
estable (0.882 → 0.863): reglas con evidencia medible y menos falsas alarmas.
Estos números se generan en cada ejecución de `scripts/optimize.py` y quedan
guardados en el JSON.

## 14. Pruebas

```bash
cd backend
python -m pytest tests/
```

Cubren: funciones de pertenencia (mínimo, máximo, punto medio, fuera de
rango), fuzzificación (pertenencias en [0,1]), reglas cargadas del artefacto
`mined_rules.json`, minería PRISM (determinista, reglas sintéticas esperadas,
discretización de términos), defuzzificación (riesgo en [0,100]), determinismo
del GA con seed fija (`p1 < p2 < p3`, dos corridas idénticas con la misma seed)
y la API (válido, faltantes, strings, fuera de rango, incoherencia térmica,
NaN).

## 15. Validación de la API

El backend nunca confía solo en el frontend: Pydantic valida tipos numéricos,
rechaza NaN/infinitos, aplica rangos basados en el dataset (config central) y
verifica la coherencia entre temperaturas (`process > air`). Errores → HTTP
422 con mensaje en español.

## 16. Tecnologías

- **Backend:** Python, FastAPI, Pydantic, pandas, NumPy, scikit-learn
  (solo para la división estratificada), pytest.
- **Lógica difusa:** scikit-fuzzy (`skfuzzy.control`).
- **Minería de reglas:** PRISM implementado en `prism.py` (pandas/NumPy).
- **Algoritmo genético:** DEAP (`creator`, `tools`).
- **Frontend:** Astro, TypeScript, HTML/CSS.

## 17. Limitaciones

- Dataset sintético (AI4I 2020); el índice de riesgo no es una probabilidad.
- Proyecto académico: no sustituye un sistema real de seguridad o
  mantenimiento.
- El umbral de decisión riesgo → fallo está fijo en 50 (configurable).
- Las reglas provienen de PRISM con umbrales de confianza/cobertura fijos; un
  umbral más alto da reglas más precisas pero con menor recall (se capturan
  menos fallos), y viceversa.

## 18. Experimento documentado

El README, el script de minería y el script de optimización responden:

1. ¿Cómo se generan las reglas? → PRISM con confianza/cobertura/lift por regla
   (`scripts/mine_rules.py` + §7).
2. ¿Cómo funciona el sistema difuso sin optimización? → §13 (columna base).
3. ¿Qué cambia después de aplicar el GA? → §13 (columna optimizada).
4. ¿Los parámetros optimizados mejoran las métricas? → sí, en todos los
   indicadores del experimento (§13).
5. ¿Cómo se comporta con datos no usados durante la optimización? → test
   aislado completo (1501 filas) reportado en §13 y en el JSON.