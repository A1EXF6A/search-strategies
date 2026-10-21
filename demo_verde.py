"""Demo comparativa: EL PICO DE ARENA (canal verde).

Muestra, lado a lado y con los MISMOS controles, dos formas de repartir la
arena que se recorta de la montaña:

- ACTUAL: reparto UNIFORME: la misma cantidad a cada una de las 256
  intensidades (es lo que hace la app ahora). Conserva la silueta: si el
  original es una colina, el resultado sigue siendo una colina.
- NUEVO: reparto TIPO AGUA: se rellenan primero las intensidades más bajas
  subiendo un "nivel de agua" hasta agotar la arena. A medida que sube el
  recorte el histograma se aplana cada vez más (en el límite, todos los bins
  a la media = ecualización).

Controles:
- Nivel de corte (% del máx.): la línea amarilla; todo lo que la supera es
  la montaña.
- Pico (100=original, 0=plano): cuánto baja la cima (curva).
- Canal: rojo / verde / azul.

Uso:
    python demo_verde.py [ruta_imagen]
    DEMO_SAVE=salida.png python demo_verde.py [ruta_imagen]   # sin ventana
"""

import os

import numpy as np
from PIL import Image

DEFAULT_IMG = "/home/alexander/Desktop/IA/pais.jpeg"
CANALES = {"rojo": 0, "verde": 1, "azul": 2}


def histograma(img, canal: int) -> np.ndarray:
    """Histograma de 256 bins del canal indicado."""
    return np.bincount(np.asarray(img)[:, :, canal].ravel(), minlength=256).astype(
        np.float64
    )


def recortar_montana(hist: np.ndarray, nivel_pct: float, pico: float):
    """Recorta la zona alta con la curva. Devuelve (h2, alto, C, arena)."""
    C = hist.max() * nivel_pct / 100.0  # nivel de corte
    alto = hist > C  # zona alta (la montaña)
    exceso = np.maximum(hist - C, 0.0)  # exceso sobre el nivel
    h2 = hist.copy()
    if pico < 100.0 and C > 0:
        k = 100.0 / max(pico, 1e-6) - 1.0  # pico=100 -> k=0 (sin recorte)
        h2[alto] = C + exceso[alto] / (1.0 + k * exceso[alto] / C)
    arena = hist[alto].sum() - h2[alto].sum()
    return h2, alto, C, arena


def reparto_uniforme(hist: np.ndarray, nivel_pct: float, pico: float):
    """ACTUAL: la arena se reparte por igual entre las 256 intensidades."""
    h2, _alto, C, arena = recortar_montana(hist, nivel_pct, pico)
    h2 = h2 + arena / 256.0
    return h2, C, arena, None


def reparto_agua(hist: np.ndarray, nivel_pct: float, pico: float):
    """NUEVO: la arena rellena primero los bins más bajos (nivel de agua)."""
    h2, _alto, C, arena = recortar_montana(hist, nivel_pct, pico)
    if arena <= 0:
        return h2, C, arena, None
    orden = np.sort(h2)
    acum = np.cumsum(orden)
    agua = float(h2.max())
    for j in range(1, 257):
        agua = (acum[j - 1] + arena) / j
        if j == 256 or orden[j] >= agua:
            break
    h2 = np.maximum(h2, agua)
    return h2, C, arena, agua


def dibujar(ax, hist, h2, C, titulo, nota):
    ax.clear()
    x = np.arange(256)
    ax.bar(x, hist, width=1.0, color="#3a3a4a", label="Original", zorder=1)
    ax.plot(x, h2, color="#6ee787", lw=1.7, label="Redistribuida", zorder=3)
    ax.axhline(C, color="#ffd166", ls="--", lw=1.4, label="Nivel de corte", zorder=2)
    ax.set_xlim(-2, 257)
    ax.set_xlabel("Intensidad (0-255)")
    ax.set_title(titulo)
    ax.legend(loc="upper right", fontsize=8)
    ax.text(
        0.015,
        0.96,
        nota,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="0.7"),
    )


def main():
    import sys

    ruta = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    salida = os.environ.get("DEMO_SAVE")  # si se define, render sin ventana

    import matplotlib

    matplotlib.use("Agg" if salida else "GTK4Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Slider

    img = np.asarray(Image.open(ruta).convert("RGB"))

    fig, (ax_u, ax_a) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    plt.subplots_adjust(bottom=0.24, top=0.9, wspace=0.08)

    ax_nivel = plt.axes([0.10, 0.13, 0.52, 0.03])
    ax_pico = plt.axes([0.10, 0.07, 0.52, 0.03])
    ax_canal = plt.axes([0.70, 0.05, 0.11, 0.12])
    s_nivel = Slider(
        ax_nivel, "Nivel de corte (% del máx.)", 5, 100, valinit=20, valstep=1
    )
    s_pico = Slider(
        ax_pico, "Pico (100=original, 0=plano)", 0, 100, valinit=40, valstep=1
    )
    radios = RadioButtons(ax_canal, tuple(CANALES), active=1)

    def actualizar(_=None):
        canal = CANALES[radios.value_selected]
        hist = histograma(img, canal)
        h_u, C, arena_u, _ = reparto_uniforme(hist, s_nivel.val, s_pico.val)
        h_a, _, arena_a, agua = reparto_agua(hist, s_nivel.val, s_pico.val)

        def nota(h2, arena, agua_v):
            lineas = [
                f"nivel   : {C:9,.0f}",
                f"arena   : {arena:9,.0f} px",
                f"agua    : {'-':>9}"
                if agua_v is None
                else f"agua    : {agua_v:9,.0f}",
                f"pico    : {hist.max():9,.0f} -> {h2.max():9,.0f}",
                f"max/min : {h2.max():9,.0f} / {h2.min():9,.0f}",
            ]
            return "\n".join(lineas)

        dibujar(
            ax_u, hist, h_u, C, "ACTUAL - reparto uniforme", nota(h_u, arena_u, None)
        )
        dibujar(
            ax_a, hist, h_a, C, "NUEVO - reparto tipo agua", nota(h_a, arena_a, agua)
        )
        ax_u.set_ylabel("Píxeles")
        ax_u.set_ylim(0, max(hist.max(), h_u.max(), h_a.max()) * 1.08)
        fig.canvas.draw_idle()

    s_nivel.on_changed(actualizar)
    s_pico.on_changed(actualizar)
    radios.on_clicked(actualizar)
    actualizar()

    if salida:
        fig.savefig(salida, dpi=110)
        print("guardado:", salida)
    else:
        plt.show()


if __name__ == "__main__":
    main()
