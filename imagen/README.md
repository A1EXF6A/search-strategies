# Procesador de imágenes RGB

Aplicación web para **modificar el brillo por canal RGB** de una imagen y **ecualizar su histograma**
(distribución de luminancia), mostrando en pantalla los histogramas **original** y **distribuido**.

- **Frontend:** Astro (sitio estático + JavaScript en el navegador).
- **Backend:** FastAPI (Python) con NumPy y Pillow, encargado de todo el procesamiento.

---

## Requisitos

| Herramienta | Versión recomendada |
|---|---|
| Node.js | 22 o superior |
| Python | 3.10 o superior |

## Cómo levantarlo

Necesitas **dos terminales**: una para el backend y otra para el frontend.

### 1. Backend (puerto 8000)

```powershell
cd backend

# Instalar dependencias (solo la primera vez)
python -m pip install -r requirements.txt

# Levantar el servidor
python -m uvicorn main:app --port 8000
```

Debe quedar escuchando en `http://127.0.0.1:8000`.

### 2. Frontend (puerto 4321)

```powershell
cd frontend

# Instalar dependencias (solo la primera vez)
npm install

# Levantar el servidor de desarrollo
npm run dev
```

Abre el navegador en `http://localhost:4321`.

> La URL del backend es configurable. Por defecto apunta a `http://127.0.0.1:8000`.
> Para cambiarla, crea un archivo `frontend/.env` con:
> `PUBLIC_API_URL=http://mi-servidor:8000`

---

## Qué hace el sistema

1. **Carga una imagen**: al abrir la página se carga automáticamente una *imagen de ejemplo* de bajo
   contraste (generada por el backend), o puedes **subir tu propia imagen** (JPG, PNG, etc.).

2. **Ajuste de brillo RGB**: tres deslizadores (**R, G, B**), cada uno de `-100` a `+100`.
   Cada valor se **suma** al canal correspondiente de cada píxel y el resultado se recorta al rango
   válido `0–255`:

   ```
   R' = clip(R + brillo_R, 0, 255)
   G' = clip(G + brillo_G, 0, 255)
   B' = clip(B + brillo_B, 0, 255)
   ```

3. **Ecualización del histograma (distribución equilibrada)**: al activar la casilla
   *“Ecualizar luminancia (distribuir histograma)”*, el sistema:

   - Calcula la **luminancia** de cada píxel: `Y = 0.299·R + 0.587·G + 0.114·B`.
   - Construye su histograma y la **función de distribución acumulada (CDF)**.
   - Remapea cada nivel de gris al rango completo `0–255` usando la CDF, **repartiendo** los valores
     de forma más uniforme.
   - Reescala R, G y B proporcionalmente para **conservar el color original** (solo cambia el brillo).

4. **Histogramas en pantalla**: se muestran dos gráficos, cada uno con las curvas de **R, G, B y
   luminancia**:

   - **Histograma original** → cómo está distribuida la imagen (con el brillo ya aplicado).
   - **Histograma distribuido (ecualizado)** → cómo queda tras la ecualización (más repartido).

   Así se puede **comparar visualmente** el antes y el después.

5. **Vista previa y descarga**: se ven las dos imágenes lado a lado (original/procesada) y el botón
   **Descargar** guarda el resultado.

---

## Estructura del proyecto

```
Deber imagen/
├── backend/
│   ├── main.py            # API FastAPI: endpoints /api/process y /api/sample
│   ├── image_utils.py     # Brillo, histogramas y ecualización (NumPy + Pillow)
│   └── requirements.txt   # Dependencias de Python
└── frontend/
    ├── src/pages/index.astro   # Interfaz completa (controles, vista previa, histogramas)
    ├── public/                 # Recursos estáticos
    ├── astro.config.mjs
    └── package.json
```

## API del backend

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/sample` | Devuelve la imagen de ejemplo (PNG de bajo contraste). |
| `POST` | `/api/process` | Recibe la imagen + valores de brillo + `equalize`. Devuelve las imágenes (antes/después) en base64 y los histogramas originales y ecualizados. |
| `GET` | `/docs` | Documentación interactiva automática de FastAPI. |

### Ejemplo de petición

```bash
curl -X POST http://127.0.0.1:8000/api/process \
  -F "file=@mi_imagen.png" \
  -F "brightness_r=20" \
  -F "brightness_g=-10" \
  -F "brightness_b=0" \
  -F "equalize=true"
```

---

## Problemas comunes

- **La página muestra un error tipo “¿Está corriendo el backend…?”**
  El backend no está levantado o la URL no coincide. Verifica que `uvicorn` esté corriendo en el
  puerto `8000`.
- **El puerto ya está en uso**
  Cambia el puerto (`uvicorn main:app --port 8001`) y ajusta `PUBLIC_API_URL` en el frontend.
- **La imagen no se procesa**
  Verifica que sea un formato soportado (JPG, PNG, etc.) y que no supere los 20 MB.
