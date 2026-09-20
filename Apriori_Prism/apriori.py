from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

RUTA_DATOS = "dataset_apriori_supermercado.csv"

path: Path = Path(__file__).parent / "dataset_apriori_supermercado.csv"

df = pd.read_csv(path)
COLUMNA_ID = "Transaccion_ID"
if COLUMNA_ID in df.columns:
    df = df.drop(columns=[COLUMNA_ID])

df = df.astype(bool)

productos = df.columns.tolist()
print(f"Transacciones: {len(df)} | Productos: {productos}")
print("\nFrecuencia de compra por producto:")
print(df.sum().sort_values(ascending=False))

MIN_SOPORTE = 0.70

itemsets_frecuentes = apriori(df, min_support=MIN_SOPORTE, use_colnames=True)
itemsets_frecuentes["longitud"] = itemsets_frecuentes["itemsets"].apply(len)

print(
    f"\nSe encontraron {len(itemsets_frecuentes)} itemsets frecuentes "
    f"(soporte >= {MIN_SOPORTE})."
)
print("\nTop 10 itemsets frecuentes por soporte:")
print(
    itemsets_frecuentes.sort_values("support", ascending=False)
    .head(10)
    .to_string(index=False)
)

MIN_CONFIANZA = 0.85

reglas = association_rules(
    itemsets_frecuentes, metric="confidence", min_threshold=MIN_CONFIANZA
)

reglas["antecedente"] = reglas["antecedents"].apply(lambda x: ", ".join(sorted(x)))
reglas["consecuente"] = reglas["consequents"].apply(lambda x: ", ".join(sorted(x)))

columnas_reporte = [
    "antecedente",
    "consecuente",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
]
reglas = reglas[columnas_reporte].round(4)

print(f"\nSe generaron {len(reglas)} reglas con confianza >= {MIN_CONFIANZA}.")

reglas_interesantes = reglas[reglas["lift"] > 1].sort_values(
    ["lift", "confidence"], ascending=False
)

print("\n" + "=" * 100)
print(f"TODAS LAS REGLAS (lift > 1), ordenadas por lift y confianza")
print("=" * 100)
for i, fila in enumerate(reglas_interesantes.itertuples(index=False), 1):
    print(
        f"{i:2d}. SI compra [{fila.antecedente}] ENTONCES compra [{fila.consecuente}]  "
        f"soporte={fila.support:.3f}  confianza={fila.confidence * 100:.1f}%  "
        f"lift={fila.lift:.2f}  leverage={fila.leverage:.4f}  "
        f"conviction={fila.conviction:.2f}"
    )

TOP_N = 10
mejores = reglas_interesantes.head(TOP_N)

print("\n" + "=" * 100)
print(f"TOP {TOP_N} MEJORES REGLAS DE ASOCIACIÓN (mayor lift)")
print("=" * 100)
for i, fila in enumerate(mejores.itertuples(index=False), 1):
    print(
        f"{i:2d}. SI compra [{fila.antecedente}] ENTONCES compra [{fila.consecuente}]  "
        f"soporte={fila.support:.3f}  confianza={fila.confidence * 100:.1f}%  "
        f"lift={fila.lift:.2f}"
    )

reglas_interesantes.to_csv("reglas_apriori.csv", index=False, encoding="utf-8-sig")
print(
    f"\nSe exportaron {len(reglas_interesantes)} reglas a 'reglas_apriori.csv'"
    f" (incluye todas con lift > 1; las primeras {TOP_N} filas son las mejores)."
)

