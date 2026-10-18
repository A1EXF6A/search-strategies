"""Demo: EL PICO DE ARENA (canal verde).

1. NIVEL DE CORTE (slider): marca la "zona alta". Todo lo que supere el nivel
   es montaña; bajando el nivel entra toda la montaña (no solo la punta).
2. PICO (slider): qué tan pronunciada queda la cima.
   - 100 -> pico original (un pincho).
   - 0   -> se aplana justo al nivel (mesa recta).
   - Medio -> recorte CURVO: los bins más altos pierden proporcionalmente más,
     así la cima baja y queda una loma suave, SIN repartir a los bins vecinos.

La arena recortada se reparte POR IGUAL entre lo que queda por debajo del
nivel (la base se infla de forma uniforme). Es O(256) por canal: barato.
"""

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("GTK4Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ------------------------------------------------------------- histograma ---
img = np.asarray(Image.open("/home/alexander/Desktop/IA/image.jpg").convert("RGB"))
canal = img[:, :, 1].ravel()
hist = np.bincount(canal, minlength=256).astype(np.float64)
MAX = hist.max()
TOTAL = hist.sum()
x = np.arange(256)


def redistribuir(nivel_pct: float, pico: float):
    """Recorta la zona alta con una curva y esparce la arena por la base."""
    C = MAX * nivel_pct / 100.0            # nivel de corte
    alto = hist > C                        # zona alta (la montaña)
    exceso = np.maximum(hist - C, 0.0)     # exceso sobre el nivel
    h2 = hist.copy()
    if pico < 100.0:
        k = 100.0 / max(pico, 1e-6) - 1.0  # pico=100 -> k=0 (sin recorte)
        h2[alto] = C + exceso[alto] / (1.0 + k * exceso[alto] / C)
    base = ~alto
    if base.any():
        arena = hist[alto].sum() - h2[alto].sum()
        h2[base] += arena / base.sum()
    return h2, alto, C


# ----------------------------------------------------------------- dibujo ---
fig, ax = plt.subplots(figsize=(11, 5.6))
plt.subplots_adjust(bottom=0.22, top=0.9)

ax.bar(x, hist, width=1.0, color="#3a3a4a", label="Original", zorder=1)
linea, = ax.plot(x, hist, color="#6ee787", lw=1.7, label="Redistribuida", zorder=3)
nivel_line = ax.axhline(0, color="#ffd166", ls="--", lw=1.4, label="Nivel de corte", zorder=2)

ax.set_xlim(-2, 257)
ax.set_xlabel("Intensidad (0-255)")
ax.set_ylabel("Píxeles")
ax.legend(loc="upper right")
titulo = ax.set_title("")

texto = ax.text(
    0.015, 0.96, "", transform=ax.transAxes, ha="left", va="top", fontsize=9,
    family="monospace", bbox=dict(boxstyle="round", fc="white", ec="0.7"),
)

ax_nivel = plt.axes([0.12, 0.11, 0.76, 0.035])
ax_pico = plt.axes([0.12, 0.05, 0.76, 0.035])
s_nivel = Slider(ax_nivel, "Nivel de corte (% del máx.)", 5, 100, valinit=20, valstep=1)
s_pico = Slider(ax_pico, "Pico (100=original, 0=plano)", 0, 100, valinit=40, valstep=1)


def actualizar(_):
    h2, alto, C = redistribuir(s_nivel.val, s_pico.val)
    base = ~alto
    linea.set_data(x, h2)
    nivel_line.set_ydata([C, C])
    ax.set_ylim(0, max(hist.max(), h2.max()) * 1.12)

    i = int(np.argmax(hist))
    vecinos = max(h2[i - 1], h2[i + 1]) if 0 < i < 255 else h2[i]
    prom = (h2[i] - vecinos) / h2[i] * 100 if h2[i] > 0 else 0.0
    arena = hist[alto].sum() - h2[alto].sum()
    sube = arena / base.sum() if base.any() else 0.0
    titulo.set_text(
        f"Montaña (bins 0-16): {hist[:17].sum():,.0f} → {h2[:17].sum():,.0f}   "
        f"·   Arena movida: {arena:,.0f} px  (+{sube:,.1f} por bin de base)"
    )
    texto.set_text(
        f"nivel de corte : {C:8,.0f}  ({s_nivel.val}% del máx)\n"
        f"bins en zona   : {alto.sum():8d}  (> nivel)\n"
        f"bins de base   : {base.sum():8d}\n"
        f"pico (bin {i:>3}) : {hist[i]:8,.0f} → {h2[i]:,.0f}\n"
        f"prominencia    : {prom:7.1f} %  (vs bins vecinos)\n"
        f"total píxeles  : {TOTAL:8,.0f} → {h2.sum():,.0f}"
    )
    fig.canvas.draw_idle()


s_nivel.on_changed(actualizar)
s_pico.on_changed(actualizar)
actualizar(None)
plt.show()
