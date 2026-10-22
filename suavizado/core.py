import random

import numpy as np
from PIL import Image


def parse_kernel(rows):
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


def apply_mask(image, kernel, use_padding=True, step=1):
    """image: PIL.Image en modo RGB. kernel: lista de listas de floats.

    Cada pixel de salida es la suma de (valor del kernel * pixel) sobre
    la vecindad. Si use_padding es True se rellena con negro el borde
    de grosor size//2 y la salida tiene el mismo tamano que la entrada;
    si es False solo se procesan las vecindades completas (salida mas
    pequena). El divisor es la suma del kernel (1 si esa suma es 0) y
    el resultado se recorta a [0, 255].

    step: salto con el que se aplica la mascara. Con 1 se procesa pixel a
    pixel (todo lo de antes); con N solo se calculan las posiciones
    0, N, 2N... de cada fila y columna, asi que la salida queda reducida
    a ceil(tamano/N). Debe ser un entero mayor o igual que 1.
    """
    if step < 1:
        raise ValueError("El salto debe ser 1 o mas.")
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

    full_w, full_h = out_w, out_h
    out_w = (full_w + step - 1) // step
    out_h = (full_h + step - 1) // step

    out = bytearray(out_w * out_h * 3)
    for oy_out in range(out_h):
        oy = oy_out * step
        row_start = oy - pad_y
        base_out = oy_out * out_w * 3
        for ox_out in range(out_w):
            ox = ox_out * step
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
            pos = base_out + ox_out * 3
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


def filter_image(image, kind, kernel=None, size=3, use_padding=True, step=1):
    """Aplica el filtro elegido y devuelve una imagen nueva.

    kind: 'mask' usa la mascara kernel (media ponderada), 'mean' calcula
    la media aritmetica de la vecindad size x size, 'median' y 'mode'
    calculan la mediana y la moda por canal de esa misma vecindad.
    use_padding significa lo mismo que en apply_mask (borde negro y
    salida del mismo tamano que la entrada) y step es el salto con el
    que se aplica el filtro (1 = pixel a pixel).
    """
    if kind == "mask":
        if kernel is None:
            raise ValueError("El filtro mascara necesita una matriz.")
        return apply_mask(image, kernel, use_padding, step)
    if kind in ("mean", "median", "mode"):
        if size < 1 or size % 2 == 0:
            raise ValueError("El tamano del filtro debe ser impar y positivo.")
        if kind == "mean":
            uniform = [[1.0] * size for _ in range(size)]
            return apply_mask(image, uniform, use_padding, step)
        return _rank_filter(image, size, use_padding, kind, step)
    raise ValueError("Filtro desconocido: %r" % kind)


def _rank_filter(image, size, use_padding, kind, step=1):
    """Mediana ('median') o moda ('mode') por canal, en Python puro.

    Recorre la imagen igual que apply_mask: con use_padding la vecindad
    se rellena de negro y la salida tiene el mismo tamano; sin padding
    solo se procesan las vecindades completas. Cada canal se resuelve por
    separado, como en los filtros estadisticos habituales. step controla
    el salto igual que en apply_mask: con N solo se calculan las
    posiciones 0, N, 2N... y la salida queda reducida a ceil(tamano/N).
    """
    if step < 1:
        raise ValueError("El salto debe ser 1 o mas.")
    image = image.convert("RGB")
    width, height = image.size
    raw = image.tobytes()
    pad = size // 2

    if not use_padding:
        if width < size or height < size:
            raise ValueError(
                "La imagen (%dx%d) es mas pequena que el filtro (%dx%d)."
                % (width, height, size, size)
            )
        out_w = width - size + 1
        out_h = height - size + 1
    else:
        out_w = width
        out_h = height

    channels = (raw[0::3], raw[1::3], raw[2::3])
    pick = _median if kind == "median" else _mode

    full_w, full_h = out_w, out_h
    out_w = (full_w + step - 1) // step
    out_h = (full_h + step - 1) // step

    out = bytearray(out_w * out_h * 3)
    for oy_out in range(out_h):
        oy = oy_out * step
        row_start = oy - pad
        base_out = oy_out * out_w * 3
        for ox_out in range(out_w):
            ox = ox_out * step
            col_start = ox - pad
            # indices de la vecindad; -1 marca el borde negro del padding
            indexes = []
            for ky in range(size):
                yy = row_start + ky
                if yy < 0 or yy >= height:
                    indexes.extend((-1,) * size)
                    continue
                base = yy * width
                for kx in range(size):
                    xx = col_start + kx
                    indexes.append(base + xx if 0 <= xx < width else -1)
            pos = base_out + ox_out * 3
            for channel, chan in enumerate(channels):
                values = [chan[i] if i >= 0 else 0 for i in indexes]
                out[pos + channel] = pick(values)

    return Image.frombytes("RGB", (out_w, out_h), bytes(out))


def _median(values):
    """Valor central de la vecindad ordenada."""
    values.sort()
    return values[len(values) // 2]


def _mode(values):
    """Valor mas frecuente de la vecindad; con empate se queda el menor."""
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return min(counts, key=lambda value: (-counts[value], value))


def imnoise(image, density=0.05, noise_type="salt & pepper", seed=None):
    """Agrega ruido de sal y pimienta al estilo de imnoise de MATLAB.

    image: PIL.Image (se convierte a RGB). density: fraccion de pixeles
    alterados, entre 0 y 1 (0.05 = 5%, el valor por defecto de MATLAB);
    de los pixeles alterados, aproximadamente la mitad pasa a 255 (sal)
    y la otra mitad a 0 (pimienta), en los tres canales del pixel, igual
    que en MATLAB. seed: semilla opcional para repetir el mismo ruido.
    Devuelve una imagen nueva, sin modificar la de entrada.
    """
    if noise_type != "salt & pepper":
        raise ValueError("Tipo de ruido no soportado: %r" % noise_type)
    if not 0.0 <= density <= 1.0:
        raise ValueError("La densidad debe estar entre 0 y 1 (ejemplo: 0.05).")

    image = image.convert("RGB")
    if density == 0:
        return image

    width, height = image.size
    out = bytearray(image.tobytes())
    rng = random.Random(seed)

    for index in range(width * height):
        if rng.random() >= density:
            continue
        value = 255 if rng.random() < 0.5 else 0
        pos = index * 3
        out[pos] = out[pos + 1] = out[pos + 2] = value

    return Image.frombytes("RGB", (width, height), bytes(out))


def frequency_filter(kind, cutoff, highpass=False, order=2, size=(1, 1)):
    """Matriz H(u,v) del filtro en frecuencias (numpy.ndarray de floats).

    kind: 'ideal', 'butterworth' o 'gaussian'. cutoff: frecuencia de
    corte D0 en ciclos por imagen (0 en el centro y ~ancho/2 o alto/2
    en el limite, el Nyquist). highpass=True usa el paso alto
    correspondiente, H = 1 - H_pasobajo. order: orden n de Butterworth.
    size: (ancho, alto) de la imagen.

    La matriz sale con el mismo orden que np.fft.fft2 (sin centrar: el
    componente DC queda en la esquina), lista para multiplicar el
    espectro sin hacer fftshift.
    """
    if kind not in ("ideal", "butterworth", "gaussian"):
        raise ValueError("Filtro de frecuencias desconocido: %r" % kind)
    if cutoff <= 0:
        raise ValueError("La frecuencia de corte debe ser mayor que 0.")
    if kind == "butterworth" and order < 1:
        raise ValueError("El orden de Butterworth debe ser 1 o mas.")
    width, height = size
    if width < 1 or height < 1:
        raise ValueError("Tamano de imagen no valido: %r" % (size,))

    # Distancia D = sqrt(u^2 + v^2) en unidades de frecuencia (ciclos),
    # con los mismos indices que devuelve fft2.
    u = np.fft.fftfreq(height) * height
    v = np.fft.fftfreq(width) * width
    du, dv = np.meshgrid(u, v, indexing="ij")
    d = np.sqrt(du * du + dv * dv)

    if kind == "ideal":
        low = np.where(d <= cutoff, 1.0, 0.0)
    elif kind == "butterworth":
        low = 1.0 / (1.0 + np.power(d / cutoff, 2 * order))
    else:
        low = np.exp(-(d * d) / (2.0 * cutoff * cutoff))

    return 1.0 - low if highpass else low


def apply_frequency_filter(image, kind="gaussian", cutoff=30.0, highpass=False, order=2):
    """Filtra en el dominio de la frecuencias: FFT -> H(u,v) -> IFFT.

    Convierte la imagen a RGB, hace la FFT de cada canal con
    np.fft.fft2, multiplica por la matriz de frequency_filter, devuelve
    con la inversa y recorta a [0, 255]. El tamano de salida es el de la
    entrada. Con paso alto los valores negativos se recortan a 0, igual
    que en los ejemplos tipicos de los libros (la imagen sale oscura,
    con las bordas iluminadas).
    """
    image = image.convert("RGB")
    width, height = image.size
    matrix = frequency_filter(kind, cutoff, highpass, order, (width, height))

    pixels = np.asarray(image, dtype=np.float64)
    filtered = np.empty_like(pixels)
    for channel in range(3):
        spectrum = np.fft.fft2(pixels[:, :, channel])
        spectrum *= matrix
        filtered[:, :, channel] = np.fft.ifft2(spectrum).real

    np.clip(filtered, 0.0, 255.0, out=filtered)
    return Image.fromarray(filtered.astype(np.uint8), "RGB")


def frequency_filter_image(kind, cutoff, highpass=False, order=2, size=(1, 1)):
    """Matriz H(u,v) centrada (DC en el centro) como imagen de grises.

    Sirve para ver la matriz del filtro: negro = 0, blanco = 1. No
    altera los valores; solo centra con fftshift para que se vea como en
    los libros.
    """
    matrix = frequency_filter(kind, cutoff, highpass, order, size)
    centered = np.fft.fftshift(matrix)
    low = float(centered.min())
    high = float(centered.max())
    span = high - low if high > low else 1.0
    pixels = ((centered - low) / span * 255.0).astype(np.uint8)
    return Image.fromarray(pixels, "L")
