"""Backend FastAPI: brillo RGB + ecualización de luminancia + histogramas."""

import io

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

import image_utils as iu

app = FastAPI(title="Procesador de imágenes RGB")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@app.get("/api/sample")
def get_sample():
    """Imagen de prueba (bajo contraste) para demostrar la ecualización."""
    img = iu.generate_sample_image()
    buf = io.BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    brightness_r: int = Form(0),
    brightness_g: int = Form(0),
    brightness_b: int = Form(0),
    equalize: bool = Form(False),
):
    """Aplica brillo RGB y, opcionalmente, ecualiza la luminancia.

    Devuelve las imágenes (antes/después) como data-URL y los histogramas
    originales y distribuidos de los canales R, G, B e Y.
    """
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="La imagen supera los 20 MB.")

    try:
        img = iu.load_image(data)
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

    for name, value in (
        ("brightness_r", brightness_r),
        ("brightness_g", brightness_g),
        ("brightness_b", brightness_b),
    ):
        if not -255 <= value <= 255:
            raise HTTPException(status_code=422, detail=f"{name} debe estar entre -255 y 255.")

    bright = iu.adjust_brightness(img, brightness_r, brightness_g, brightness_b)
    equalized = iu.equalize_luminance(bright) if equalize else bright.copy()

    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "image_before": iu.to_data_url(bright),
        "image_after": iu.to_data_url(equalized),
        "hist_original": iu.compute_histograms(bright),
        "hist_equalized": iu.compute_histograms(equalized),
        "equalized": equalize,
    }
