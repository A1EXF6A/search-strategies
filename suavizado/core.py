"""Aplicacion manual de una mascara NxN sobre una imagen.

Solo se usa Pillow para leer/escribir el archivo; la convolucion
se calcula a mano con bucles puros.
"""

from PIL import Image


def parse_kernel(rows):
    """Convierte textos de la interfaz en una matriz de floats."""
    kernel = []
    for row in rows:
        line = []
        for text in row:
            text = str(text).strip()
            if not text:
                raise ValueError("La mascara tiene casillas vacias.")
            try:
                line.append(float(text.replace(",", ".")))
            except ValueError:
                raise ValueError("Valor no numerico: %r" % text)
        kernel.append(line)
    if not kernel:
        raise ValueError("La mascara esta vacia.")
    width = len(kernel[0])
    for line in kernel:
        if len(line) != width:
            raise ValueError("La mascara debe ser cuadrada.")
    return kernel


def kernel_sum(kernel):
    return sum(sum(line) for line in kernel)


def apply_mask(image, kernel, use_padding=True):
    """image: PIL.Image en modo RGB. kernel: lista de listas de floats.

    Cada pixel de salida es la suma de (valor del kernel * pixel) sobre
    la vecindad. Si use_padding es True se rellena con negro el borde
    de grosor size//2 y la salida tiene el mismo tamano que la entrada;
    si es False solo se procesan las vecindades completas (salida mas
    pequena). El divisor es la suma del kernel (1 si esa suma es 0) y
    el resultado se recorta a [0, 255].
    """
    image = image.convert("RGB")
    width, height = image.size
    raw = image.tobytes()

    kh = len(kernel)
    kw = len(kernel[0])
    pad_y = kh // 2
    pad_x = kw // 2

    if not use_padding:
        if width < kw or height < kh:
            raise ValueError(
                "La imagen (%dx%d) es mas pequena que la mascara (%dx%d)."
                % (width, height, kw, kh)
            )
        out_w = width - kw + 1
        out_h = height - kh + 1
    else:
        out_w = width
        out_h = height

    reds = raw[0::3]
    greens = raw[1::3]
    blues = raw[2::3]

    total = kernel_sum(kernel)
    divisor = total if total != 0 else 1.0

    out = bytearray(out_w * out_h * 3)
    for oy in range(out_h):
        row_start = oy - pad_y
        base_out = oy * out_w * 3
        for ox in range(out_w):
            col_start = ox - pad_x
            sr = sg = sb = 0.0
            for ky in range(kh):
                yy = row_start + ky
                if yy < 0 or yy >= height:
                    continue
                base = yy * width
                krow = kernel[ky]
                for kx in range(kw):
                    k = krow[kx]
                    if k == 0:
                        continue
                    xx = col_start + kx
                    if xx < 0 or xx >= width:
                        continue
                    idx = base + xx
                    sr += k * reds[idx]
                    sg += k * greens[idx]
                    sb += k * blues[idx]
            pos = base_out + ox * 3
            out[pos] = _clamp(sr / divisor)
            out[pos + 1] = _clamp(sg / divisor)
            out[pos + 2] = _clamp(sb / divisor)

    return Image.frombytes("RGB", (out_w, out_h), bytes(out))


def _clamp(value):
    if value < 0:
        return 0
    if value > 255:
        return 255
    return int(value + 0.5)
