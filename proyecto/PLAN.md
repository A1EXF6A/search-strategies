# Planificación — Sistema inteligente de mantenimiento preventivo con lógica difusa + algoritmo genético

> Documento de referencia del proyecto. Contiene el plan completo tal como lo definió el usuario.
> La implementación sigue el orden de la sección "46. Orden de implementación".

## 1. Objetivo general

Desarrollar una aplicación web académica de mantenimiento preventivo para una máquina industrial de manufactura con elementos rotativos.

El sistema permitirá que un usuario introduzca las condiciones actuales de operación de una máquina y recibirá:

- un **índice de riesgo de fallo de 0 a 100**;
- un **nivel de riesgo**;
- una **acción recomendada de mantenimiento**;
- una explicación sencilla de los factores que contribuyeron al resultado.

La parte inteligente del sistema estará compuesta por dos técnicas de Inteligencia Artificial:

1. **Lógica difusa:** será el corazón del sistema de inferencia y tomará la decisión a partir de las variables de entrada.
2. **Algoritmo genético:** optimizará los parámetros de las funciones de pertenencia utilizadas por el sistema difuso.

El dataset será utilizado exclusivamente en el backend para entrenar/optimizar el sistema y evaluar su comportamiento. El usuario final no interactuará directamente con el dataset.

El dataset de referencia será el **AI4I 2020 Predictive Maintenance Dataset de UCI**, que es sintético, contiene 10.000 observaciones y representa datos de mantenimiento predictivo industrial.

## 2. Principio fundamental de arquitectura

El proyecto debe ser **intencionalmente sencillo**. No implementar microservicios, arquitectura hexagonal, DDD, repositorios innecesarios, bases de datos, ORM, autenticación, WebSockets, colas, múltiples servicios backend, contenedores Docker (salvo convención existente), ni capas innecesarias de abstracción.

Separación clara:

```text
frontend/
    Interfaz del usuario

backend/
    Procesamiento
    Lógica difusa
    Algoritmo genético
    Dataset
    API
```

Flujo principal:

```text
Usuario → Formulario → API → Validación → Variables derivadas → Fuzzificación
→ Reglas difusas → Agregación → Defuzzificación → Índice de riesgo
→ Acción recomendada → Respuesta al frontend
```

El algoritmo genético **no debe ejecutarse en cada consulta del usuario**. Funciona así:

```text
Dataset → Entrenamiento/optimización → GA → Parámetros óptimos
→ Archivo de parámetros → Sistema difuso
```

## 3. Tecnología propuesta

- **Frontend:** Astro + TypeScript + HTML/CSS + JS/TS para interacción. Comunicación por HTTP/JSON.
- **Backend:** Python + FastAPI + pandas + NumPy (+ scikit-learn solo para la división del dataset). La lógica difusa se implementa con **scikit-fuzzy** (`skfuzzy.control`: Antecedent/Consequent/Rule/ControlSystem/ControlSystemSimulation) y el algoritmo genético con **DEAP** (`creator`, `tools`). El usuario eligió estas librerías en lugar de una implementación manual (decisión explícita: usar librerías para la fuzzificación y el GA "para todo").
- Las convenciones de los proyectos existentes del workspace (focos, Apriori_Prism) tienen prioridad: identificadores en inglés, mensajes de consola en español, Python tipado, pyright 0 errores, frontend Astro + pnpm.

## 4. Regla respecto a proyectos anteriores

1. Inspeccionar las carpetas de los demás proyectos.
2. Identificar convenciones de nombres, estructura, formato, imports, configuración, naming, manejo de errores, README, comandos, componentes frontend.
3. Reutilizar las convenciones existentes cuando sean compatibles.

Prioridad: `convenciones existentes → simplicidad → claridad → funcionalidad`.

## 5. Datos de entrada del usuario

El formulario solicita exactamente cinco variables (documentadas en el dataset AI4I 2020):

| Variable | Unidad |
| -------- | ------ |
| Air Temperature | K |
| Process Temperature | K |
| Rotational Speed | rpm |
| Torque | Nm |
| Tool Wear | min |

## 6. Variable derivada

```text
thermal_gap = process_temperature - air_temperature
```

NO se solicita al usuario. Aparece opcionalmente como información calculada. Entradas del usuario: 5. Variables utilizadas internamente: 5 + 1 derivada.

## 7. Resultado del sistema

Índice de riesgo: número 0-100 llamado **"Índice de riesgo"** (no "probabilidad de fallo").

## 8. Niveles de riesgo

| Rango | Nivel |
| ----- | ----- |
| 0–24 | Bajo |
| 25–49 | Moderado |
| 50–74 | Alto |
| 75–100 | Crítico |

Los límites se centralizan en una constante de configuración (config.py).

## 9. Acción recomendada

| Riesgo | Acción |
| ------ | ------ |
| Bajo | Continuar operación |
| Moderado | Monitorear |
| Alto | Programar mantenimiento |
| Crítico | Detener y revisar |

## 10. Explicación del resultado

Provenir de las reglas difusas activadas (no texto fijo inventado).

## 11. Lógica difusa

Enfoque **Mamdani**: fuzzificación → grados de pertenencia → evaluación de reglas → agregación → defuzzificación → índice.

## 12. Variables lingüísticas

Tres conjuntos difusos `LOW / MEDIUM / HIGH` para: Process Temperature, Thermal Gap, Rotational Speed, Torque, Tool Wear. (Air Temperature NO se fuzzifica: solo alimenta thermal_gap.)

## 13. Funciones de pertenencia

Trapezoidales (LOW, HIGH) y triangulares (MEDIUM), definidas con
`skfuzzy.trapmf` / `skfuzzy.trimf`. `trapmf` valida `a ≤ b ≤ c ≤ d`, por lo que
el decodificador del GA garantiza el orden y la separación mínima de los
puntos de corte.

## 14. Algoritmo genético

Optimiza los parámetros de las funciones de pertenencia. NO genera el resultado por usuario. NO reemplaza la lógica difusa.

## 15. Qué optimiza el GA

Cada variable se representa con tres puntos de corte `p1 < p2 < p3`:

```text
LOW    = [min, min, p1, p2]
MEDIUM = [p1, p2, p3]
HIGH   = [p2, p3, max, max]
```

Cinco variables → **15 genes**. Genes normalizados 0..1 → transformados al rango real. Al decodificar, ordenar los tres puntos y aplicar separación mínima.

## 16. Fitness del GA

Proceso: individuo → parámetros de membresías → sistema difuso → procesar datos de entrenamiento → calcular riesgo → convertir a fallo/no fallo → comparar con `Machine Failure`.

Fitness = **Balanced Accuracy** (por desbalance de clases). Métricas adicionales: Accuracy, Precision, Recall, F1, Confusion Matrix.

Para acotar el costo de cada evaluación, el fitness se calcula sobre una **muestra estratificada** de 600 filas (todos los positivos + negativos al azar, seed fija; `FITNESS_SAMPLE_SIZE` en config). Las métricas finales se reportan siempre sobre el conjunto (train/test) completo.

## 17. División del dataset

70% entrenamiento, 15% validación, 15% test (el test queda aislado). Preferir división estratificada reproducible porque el orden de filas no representa secuencia temporal útil (documentado en README).

## 18. Funcionamiento del GA

Tradicional: población inicial → fitness → selección → crossover → mutación → nueva generación → elitismo → repetir → guardar mejor. Configuración sugerida: `population_size=30, generations=50, elite_count=2, crossover_rate≈0.8, mutation_rate≈0.1, random_seed=fijo`. Valores configurables.

## 19. Resultado del entrenamiento

`backend/artifacts/optimized_parameters.json` con: parámetros optimizados, configuración de membresías, seed, fitness, versión, fecha opcional. No guardar el dataset.

## 20. Comparación antes y después del GA

Métricas del sistema sin optimizar vs optimizado (Balanced Accuracy, Precision, Recall, F1) sobre el mismo test. Sección "Resultados de optimización" en el README.

## 21. Reglas difusas

Generadas por el algoritmo **PRISM** (inducción de reglas, Cendrowska, 1987) a partir del dataset AI4I 2020. El dataset se discretiza tomando el término de máxima pertenencia por variable (partición canónica = parámetros por defecto) y PRISM descubre reglas `SI variable = término ENTONCES fallo/sin fallo` con métricas medibles (confianza, cobertura, lift). Umbrales: fallo con confianza ≥ 0.40 y cobertura ≥ 5; seguras con confianza ≥ 0.965 y cobertura ≥ 50. Los consecuentes son `high`/`critical` para reglas de fallo y `low` para seguras.

- Las reglas se guardan en `backend/artifacts/mined_rules.json` (fuente de verdad).
- `backend/config.py` carga las reglas desde el artefacto (`FUZZY_RULES`); NO van dispersas en main.py, NO se editan a mano.
- La membresía (p1/p2/p3) la optimiza el GA; las reglas son fijas y reproducibles (el script de minería es determinista).

## 22. Salida difusa

Variable de salida `risk` en 0..100 con cuatro conjuntos: LOW, MEDIUM, HIGH, CRITICAL. Defuzzificación por **centroide**.

## 23. Activación de reglas

AND → `min(...)`. Combinación de reglas → `max(...)` (mecanismo por defecto de `skfuzzy.control.Rule`).

## 24. Explicación interna

El motor devuelve `risk_score`, `risk_level`, `action`, `activated_rules` (nombre + fuerza) y membresías. El frontend genera la explicación.

## 25. API del backend

- `GET /api/health` → `{"status": "ok"}`
- `POST /api/predict` → `{risk_score, risk_level, action, thermal_gap, factors, activated_rules}` (nombres en snake_case según convención).

## 26. Validación del backend

Validar tipos numéricos, faltantes, NaN, infinitos, rangos, coherencia de temperaturas. Usar Pydantic mediante FastAPI. Límites basados en rangos del dataset (config central).

## 27. Flujo del backend

`main.py` pequeño:

```python
# 1. Cargar parámetros optimizados
# 2. Crear sistema difuso
# 3. Crear aplicación FastAPI
# 4. Definir endpoints
```

No colocar GA, membresías, reglas, CSV, métricas ni entrenamiento dentro de main.py.

## 28. Estructura del backend

```text
backend/
├── data/ai4i2020.csv
├── artifacts/optimized_parameters.json
├── artifacts/mined_rules.json   (base de reglas generada por PRISM)
├── main.py            (API y arranque)
├── config.py          (rangos, niveles, acciones, cargador de reglas, parámetros por defecto)
├── fuzzy.py           (sistema difuso Mamdani con scikit-fuzzy: membresías, inferencia, agregación, defuzzificación)
├── prism.py           (minería de reglas PRISM: discretización, inducción de reglas, métricas)
├── genetic.py         (algoritmo genético con DEAP: individuo, población, fitness, selección, crossover, mutación, elitismo)
├── model.py           (cargar parámetros, construir sistema, predecir, transformar riesgo a nivel/acción)
├── schemas.py         (modelos request/response de FastAPI)
├── scripts/mine_rules.py (CSV → discretización → PRISM → mined_rules.json → reporte)
├── scripts/optimize.py (CSV → split → GA → evaluación → guardar parámetros → métricas)
├── tests/test_fuzzy.py
├── requirements.txt
└── README.md
```

## 29. Flujo de ejecución

- **Fase A — Minería de reglas:** `python -m scripts.mine_rules` → genera `mined_rules.json` (base difusa con confianza, cobertura y lift por regla).
- **Fase B — Optimización:** `python -m scripts.optimize` → genera `optimized_parameters.json`.
- **Fase C — Aplicación:** `uvicorn main:app` → carga JSONs, construye sistema difuso, espera requests.

## 30. Si falta un artefacto

Errores claros: "No existen parámetros optimizados. Ejecute el proceso de entrenamiento antes de iniciar el sistema." y "No existe mined_rules.json. Ejecute el proceso de minería de reglas antes de iniciar el sistema." No ejecutar GA ni minería silenciosamente en el arranque.

## 31. Frontend

Una sola experiencia (una página): formulario 5 inputs → [Analizar] → resultado con índice, nivel, acción, factores, botón [Ver detalles].

## 32. Frontend por pasos

Paso 1 formulario → Paso 2 validar inputs → Paso 3 enviar request → Paso 4 estado de procesamiento → Paso 5 resultado → Paso 6 explicación técnica opcional.

## 33. Componentes visuales

Header, Form, ResultCard, RiskIndicator, FactorsList, TechnicalDetails (reducir si la convención lo pide).

## 34. Indicador visual del riesgo

Barra de progreso + texto del nivel siempre visible (BAJO / MODERADO / ALTO / CRÍTICO). No depender solo de color.

## 35. Detalles técnicos opcionales

Sección "Detalles de la inferencia": thermal gap y membresías por variable (LOW/MEDIUM/HIGH). Útil para demostración académica.

## 36. Evitar falsos reclamos

README e interfaz aclaran: dataset sintético, proyecto académico, índice = salida de sistema difuso (no probabilidad), no sistema certificado.

## 37. NO reconstruir las reglas del dataset

No hacer `if tool_wear > 200: failure = 1`. El sistema hace fuzzificación → reglas difusas → riesgo continuo → acción.

## 38. Pruebas mínimas del backend

Membresías (min, max, punto medio, fuera de rango), fuzzificación (0<=μ<=1), reglas cargadas del artefacto, minería PRISM (determinista, reglas sintéticas), defuzzificación (0<=risk<=100), API (válidos, faltantes, strings, fuera de rango), GA (determinista con seed, población/genes válidos, p1<p2<p3, fitness reproducible).

## 39. Pruebas del frontend

Formulario vacío, input inválido, input válido, loading, error del backend, resultado correcto.

## 40. README

Explicar: nombre, problema, objetivo, dataset, tecnologías, arquitectura, variables, variable derivada, lógica difusa, reglas, GA, fitness, entrenamiento, cómo ejecutar (optimización + frontend/backend), resultados, limitaciones. Incluir referencia oficial UCI (DOI/licencia).

## 41. Experimento documentado

Preguntas: ¿cómo funciona sin optimización? ¿qué cambia con GA? ¿mejora las métricas? ¿cómo se comporta con datos no usados?

## 42. Responsabilidades de cada técnica

- **Lógica difusa:** representación de incertidumbre, fuzzificación, reglas, inferencia, decisión. "¿Qué nivel de riesgo?"
- **PRISM:** minería de reglas desde los datos. "¿Qué combinaciones de condiciones predicen el fallo y con qué evidencia?"
- **GA:** optimización. "¿Qué parámetros hacen mejor al sistema frente a los datos?"

## 43. Criterios de terminado

Checklist: frontend con 5 variables, validación en ambos lados, thermal_gap, fuzzificación, membresías, reglas generadas por PRISM (minería reproducible, métricas confianza/cobertura/lift por regla, artefacto mined_rules.json), Mamdani, centroide, riesgo 0-100, nivel, acción, factores, GA optimiza membresías, fitness con datos, split 70/15/15, archivo de parámetros, sistema sin GA por request, comparación antes/después, pruebas, README, respeta convenciones, sin complejidad innecesaria.

## 44. Prioridad final

`CLARIDAD > COMPLEJIDAD`. El main pequeño, pocas piezas, cada archivo con responsabilidad clara, todo explicable de principio a fin.