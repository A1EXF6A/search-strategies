"""Deteccion de objetos con skimage.measure.regionprops.

Pipeline: umbral fijo -> mascara binaria -> etiquetado -> regionprops.
De cada region se extrae el centroide y el tamano (bbox y area).
"""

import numpy as np
from PIL import Image, ImageDraw
from skimage import measure

COLOR_BORDE = (255, 40, 40)
COLOR_CENTRO = (0, 220, 60)
GROSOR = 2
CRUZ = 7


def mascara(gris: np.ndarray, umbral: int, invertir: bool) -> np.ndarray:
    """Objetos oscuros sobre fondo claro; invertir para el caso contrario."""
    return gris > umbral if invertir else gris < umbral


def detectar(imagen: Image.Image, umbral: int, invertir: bool = False,
             area_min: int = 1) -> list:
    """Devuelve una lista de dicts con centro, tamano y bbox de cada objeto."""
    gris = np.asarray(imagen.convert("L"))
    binaria = mascara(gris, umbral, invertir)
    etiquetas = measure.label(binaria, connectivity=2)

    objetos = []
    for region in measure.regionprops(etiquetas):
        if region.area < area_min:
            continue
        minr, minc, maxr, maxc = region.bbox
        cy, cx = region.centroid
        objetos.append({
            "centro": (float(round(cx, 1)), float(round(cy, 1))),
            "ancho": maxc - minc,
            "alto": maxr - minr,
            "area": int(region.area),
            "bbox": (minc, minr, maxc, maxr),
        })
    objetos.sort(key=lambda o: (o["centro"][1], o["centro"][0]))
    return objetos


def dibujar(imagen: Image.Image, objetos: list) -> Image.Image:
    """Copia de la imagen con rectangulos y cruces sobre cada objeto."""
    salida = imagen.convert("RGB").copy()
    lienzo = ImageDraw.Draw(salida)
    for obj in objetos:
        minc, minr, maxc, maxr = obj["bbox"]
        lienzo.rectangle([minc, minr, maxc - 1, maxr - 1],
                         outline=COLOR_BORDE, width=GROSOR)
        cx, cy = obj["centro"]
        cx, cy = int(round(cx)), int(round(cy))
        lienzo.line([cx - CRUZ, cy, cx + CRUZ, cy], fill=COLOR_CENTRO,
                    width=GROSOR)
        lienzo.line([cx, cy - CRUZ, cx, cy + CRUZ], fill=COLOR_CENTRO,
                    width=GROSOR)
    return salida
