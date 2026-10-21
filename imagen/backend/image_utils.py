"""Procesamiento de imágenes: brillo RGB, histogramas y ecualización de luminancia."""

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

LUMA_COEFFS = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def load_image(data: bytes) -> np.ndarray:
    """Decodifica bytes (JPG/PNG/...) a un arreglo RGB uint8 de shape (H, W, 3)."""
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def to_data_url(img: np.ndarray, fmt: str = "PNG") -> str:
    """Convierte un arreglo RGB uint8 a data-URL (base64) listo para <img>/canvas."""
    pil = Image.fromarray(img, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def adjust_brightness(img: np.ndarray, r: int, g: int, b: int) -> np.ndarray:
    """Suma un desplazamiento (-255..255) a cada canal R, G, B con recorte a [0, 255]."""
    offsets = np.array([r, g, b], dtype=np.int16)
    return np.clip(img.astype(np.int16) + offsets, 0, 255).astype(np.uint8)


def luminance(img: np.ndarray) -> np.ndarray:
    """Calcula el canal de luminancia Y = 0.299R + 0.587G + 0.114B (float64)."""
    return img.astype(np.float64) @ LUMA_COEFFS


def compute_histograms(img: np.ndarray) -> dict:
    """Histogramas de 256 bins para los canales R, G, B y la luminancia Y."""
    hist = {}
    for i, name in enumerate(("r", "g", "b")):
        hist[name] = np.bincount(img[:, :, i].ravel(), minlength=256).tolist()
    y_hist, _ = np.histogram(luminance(img), bins=256, range=(0, 256))
    hist["y"] = y_hist.astype(int).tolist()
    return hist


def equalize_luminance(img: np.ndarray) -> np.ndarray:
    """Ecualiza la luminancia Y mediante el mapeo CDF y reescala R, G, B
    proporcionalmente (preserva el tono original de los colores)."""
    y = luminance(img)
    y_round = np.clip(np.round(y), 0, 255).astype(np.uint8)

    hist, _ = np.histogram(y_round, bins=256, range=(0, 256))
    cdf = hist.cumsum()
    n_pixels = cdf[-1]

    cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0
    denom = n_pixels - cdf_min
    if denom <= 0:
        return img.copy()

    # LUT: nivel de entrada -> nivel ecualizado
    lut = np.floor((cdf - cdf_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
    y_eq = lut[y_round].astype(np.float64)

    # Reescala cada canal manteniendo la relación R:G:B del píxel original
    scale = y_eq / np.maximum(y, 1e-6)
    out = np.clip(img.astype(np.float64) * scale[:, :, None], 0, 255).astype(np.uint8)
    return out


def generate_sample_image(width: int = 480, height: int = 360) -> np.ndarray:
    """Genera una imagen de prueba con bajo contraste (ideal para demostrar la ecualización)."""
    y = np.linspace(0, 1, height, dtype=np.float64)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float64)[None, :]

    r = 70 + 40 * y + 20 * x
    g = 90 + 30 * y + 15 * (1 - x)
    b = 110 + 25 * (1 - y) + 0 * x
    img = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)

    pil = Image.fromarray(img, mode="RGB")
    draw = ImageDraw.Draw(pil)

    draw.ellipse((330, 40, 400, 110), fill=(160, 150, 120))
    draw.polygon([(0, 280), (140, 160), (280, 280)], fill=(85, 95, 100))
    draw.polygon([(180, 280), (320, 180), (460, 280)], fill=(100, 105, 110))
    draw.rectangle((0, 280, 480, 360), fill=(75, 85, 80))
    draw.ellipse((60, 300, 180, 345), fill=(110, 100, 90))

    return np.asarray(pil, dtype=np.uint8)
