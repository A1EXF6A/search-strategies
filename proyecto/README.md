# Sistema inteligente de mantenimiento preventivo

Aplicación web académica de **mantenimiento predictivo** con **lógica difusa
(Mamdani)** + **minería de reglas (PRISM)** + **algoritmo genético**.

```
Usuario → 5 condiciones de operación
   → API (validación)
   → thermal_gap (variable derivada)
   → Fuzzificación → Reglas difusas → Agregación → Defuzzificación (centroide)
   → Índice de riesgo (0–100) → Nivel → Acción recomendada → Factores
```

## Estructura

```
backend/    API FastAPI, lógica difusa, minería de reglas PRISM, algoritmo
            genético, dataset y artefactos (parámetros y reglas minadas)
frontend/   Interfaz de usuario (Astro, una página)
PLAN.md     Planificación completa del proyecto (documento de referencia)
```

## Documentación

- **Backend** (arquitectura, lógica difusa, GA, resultados experimentales y
  cómo ejecutar): [`backend/README.md`](backend/README.md)
- **Frontend** (cómo ejecutar la interfaz): [`frontend/README.md`](frontend/README.md)

## Resumen de arranque

```bash
# 1) Minería de reglas (una sola vez): genera artifacts/mined_rules.json
cd backend && python -m scripts.mine_rules

# 2) Optimización (una sola vez): genera artifacts/optimized_parameters.json
python -m scripts.optimize

# 3) API
uvicorn main:app --host localhost --port 8001

# 4) Interfaz (otra terminal)
cd frontend && pnpm install && pnpm dev   # http://localhost:4321
```

> Proyecto académico sobre el dataset **sintético** AI4I 2020 (UCI,
> DOI 10.24432/C5H30F). El índice de riesgo es la salida de un sistema difuso,
> no una probabilidad estadística ni un sistema industrial certificado.