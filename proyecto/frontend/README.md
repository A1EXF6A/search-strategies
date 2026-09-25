# Frontend — Mantenimiento preventivo inteligente

Interfaz de una sola página construida con **Astro** que permite introducir las
cinco condiciones de operación de la máquina y muestra el resultado de la
inferencia difusa.

## Requisitos

- Node.js >= 22.12
- pnpm

## Ejecutar

```bash
pnpm install
pnpm dev
```

El servidor de desarrollo queda en `http://localhost:4321`.

## Backend

El frontend consume la API del backend en `http://localhost:8001/api/predict`
(ver `../backend/README.md`). El backend debe estar corriendo:

```bash
cd ../backend
uvicorn main:app --host localhost --port 8001
```

## Estructura

```
src/pages/index.astro   Página única: formulario, resultado, factores y
                        detalles de la inferencia (membresías y reglas)
```

## Verificación

```bash
pnpm astro check   # type-check de los archivos Astro (0 errores)
pnpm build         # build de producción
```