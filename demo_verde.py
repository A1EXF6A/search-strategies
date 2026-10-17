"""Demo: EL PICO DE ARENA (canal verde).

Antes la "zona de corte" estaba fija en el 45 % del máximo, así que solo caía
la punta del pico y la "montaña" quedaba intacta. Ahora el NIVEL DE CORTE es
ajustable: bajándolo entra toda la montaña (no solo el pico) y se aplana.

1. ZONA ALTA = intensidades cuya cuenta supera el NIVEL DE CORTE (slider).
   - Nivel alto  -> solo el pico.
   - Nivel bajo  -> toda la montaña.
2. FUERZA (slider): qué fracción del exceso sobre el nivel se corta.
3. La arena cortada se reparte POR IGUAL entre TODAS las intensidades que
   quedan por debajo del nivel: la base se infla de forma uniforme y el total
   de píxeles se conserva.
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
MEDIA = hist.mean()          # nivel donde el histograma quedaría plano
TOTAL = hist.sum()


def redistribuir(nivel_pct: float, fuerza_pct: float):
    """Corta el exceso sobre el nivel y reparte la arena por igual en la base."""
    C = MAX * nivel_pct / 100.0            # nivel de corte en cuentas
    alto = hist > C                        # zona alta (la montaña)
    exceso = np.maximum(hist - C, 0.0)     # arena disponible sobre el nivel
    quitar = (fuerza_pct / 100.0) * exceso # cuánta arena se desprende
    h2 = hist - quitar
    base = ~alto
    if base.any():
        h2[base] += quitar.sum() / base.sum()
    return h2, alto, C


# ----------------------------------------------------------------- dibujo ---
x = np.arange(256)
fig, ax = plt.subplots(figsize=(11, 5.5))
plt.subplots_adjust(bottom=0.22, top=0.9)

barras = ax.bar(x, hist, width=1.0, color="#3a3a4a", label="Original", zorder=1)
linea, = ax.plot(x, hist, color="#6ee787", lw=1.6, label="Redistribuida", zorder=3)
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

ax_nivel = plt.axes([0.12, 0.10, 0.76, 0.035])
ax_fuerza = plt.axes([0.12, 0.045, 0.76, 0.035])
s_nivel = Slider(ax_nivel, "Nivel de corte (% del máx.)", 5, 100, valinit=20, valstep=1)
s_fuerza = Slider(ax_fuerza, "Fuerza de arena %", 0, 100, valinit=100, valstep=1)


def actualizar(_):
    h2, alto, C = redistribuir(s_nivel.val, s_fuerza.val)
    base = ~alto
    linea.set_data(x, h2)
    nivel_line.set_ydata([C, C])
    ymax = max(hist.max(), h2.max()) * 1.12
    ax.set_ylim(0, ymax)

    arena = (s_fuerza.val / 100.0) * np.maximum(hist - C, 0).sum()
    sube = arena / base.sum() if base.any() else 0.0
    titulo.set_text(
        f"Montaña (bins 0-16): {hist[:17].sum():,.0f} → {h2[:17].sum():,.0f}   "
        f"·   Arena movida: {arena:,.0f} px  (+{sube:,.1f} a cada uno de los {base.sum()} bins de base)"
    )
    texto.set_text(
        f"nivel de corte : {C:8,.0f}  ({s_nivel.val}% del máx)\n"
        f"bins en zona   : {alto.sum():8d}  (> nivel)\n"
        f"bins de base   : {base.sum():8d}\n"
        f"máx del pico   : {hist.max():8,.0f} → {h2.max():,.0f}\n"
        f"total píxeles  : {TOTAL:8,.0f} → {h2.sum():,.0f}"
    )
    fig.canvas.draw_idle()


s_nivel.on_changed(actualizar)
s_fuerza.on_changed(actualizar)
actualizar(None)
plt.show()
