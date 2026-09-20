from collections import Counter

import pandas as pd

RUTA_DATOS = "dataset_prism_proyectos_software.csv"
COLUMNA_ID = "Proyecto_ID"
COLUMNA_CLASE = "Exito_Proyecto"

df = pd.read_csv(RUTA_DATOS)
if COLUMNA_ID in df.columns:
    df = df.drop(columns=[COLUMNA_ID])

atributos = [c for c in df.columns if c != COLUMNA_CLASE]
print(f"Filas: {len(df)} | Atributos: {atributos} | Clase: {COLUMNA_CLASE}")
print(f"Distribución de clase:\n{df[COLUMNA_CLASE].value_counts()}\n")


def prism(data: pd.DataFrame, atributos: list, clase: str, soporte_minimo: int = 5):

    reglas = []
    valores_clase = data[clase].unique()

    for valor_clase in valores_clase:
        E = data.copy()
        total_clase_original = (data[clase] == valor_clase).sum()

        while (E[clase] == valor_clase).sum() > 0:
            condiciones = []
            subconjunto = E.copy()
            atributos_disponibles = list(atributos)

            while True:
                mejor_atributo = None
                mejor_valor = None
                mejor_p = -1
                mejor_t = -1
                mejor_score = -1.0

                for atributo in atributos_disponibles:
                    if atributo in [c[0] for c in condiciones]:
                        continue
                    for valor in subconjunto[atributo].unique():
                        cubiertas = subconjunto[subconjunto[atributo] == valor]
                        t = len(cubiertas)
                        p = (cubiertas[clase] == valor_clase).sum()
                        if t == 0:
                            continue
                        score = p / t
                        if (score > mejor_score) or (
                            score == mejor_score and p > mejor_p
                        ):
                            mejor_score = score
                            mejor_p = p
                            mejor_t = t
                            mejor_atributo = atributo
                            mejor_valor = valor

                if mejor_atributo is None:
                    break

                condiciones.append((mejor_atributo, mejor_valor))
                subconjunto = subconjunto[subconjunto[mejor_atributo] == mejor_valor]

                if mejor_score == 1.0:
                    break
                if len(condiciones) == len(atributos):
                    break
                if mejor_t <= soporte_minimo:
                    break

            if not condiciones:
                break

            mask = pd.Series(True, index=data.index)
            for atributo, valor in condiciones:
                mask &= data[atributo] == valor
            cubiertas_total = data[mask]
            t_total = len(cubiertas_total)
            p_total = (cubiertas_total[clase] == valor_clase).sum()
            precision = p_total / t_total if t_total else 0
            cobertura = p_total / total_clase_original if total_clase_original else 0

            reglas.append(
                {
                    "condiciones": condiciones,
                    "clase": valor_clase,
                    "precision": round(precision, 4),
                    "soporte": t_total,
                    "aciertos": p_total,
                    "cobertura_clase": round(cobertura, 4),
                }
            )

            mask_E = pd.Series(True, index=E.index)
            for atributo, valor in condiciones:
                mask_E &= E[atributo] == valor
            E = E[~mask_E]

            if p_total == 0:
                break

    return reglas


def formatear_regla(regla: dict) -> str:
    condiciones_txt = " Y ".join(f"({a} = {v})" for a, v in regla["condiciones"])
    return (
        f"SI {condiciones_txt} ENTONCES {COLUMNA_CLASE} = {regla['clase']}  "
        f"[precisión={regla['precision'] * 100:.1f}%, "
        f"soporte={regla['soporte']}, aciertos={regla['aciertos']}, "
        f"cobertura de la clase={regla['cobertura_clase'] * 100:.1f}%]"
    )


reglas = prism(df, atributos, COLUMNA_CLASE, soporte_minimo=5)

reglas_ordenadas = sorted(
    reglas, key=lambda r: (r["precision"], r["soporte"]), reverse=True
)

print("=" * 90)
print("TODAS LAS REGLAS GENERADAS POR PRISM (ordenadas por precisión y soporte)")
print("=" * 90)
for i, r in enumerate(reglas_ordenadas, 1):
    print(f"{i:2d}. {formatear_regla(r)}")

total_filas = len(df)
MIN_SOPORTE = 0.70
MIN_CONFIANZA = 0.85

mejores = [
    r
    for r in reglas_ordenadas
    if r["precision"] >= MIN_CONFIANZA and (r["soporte"] / total_filas) >= MIN_SOPORTE
]

print("\n" + "=" * 90)
print(
    f"MEJORES REGLAS (soporte >= {MIN_SOPORTE} del total y confianza >= {MIN_CONFIANZA})"
)
print("=" * 90)
if mejores:
    for i, r in enumerate(mejores, 1):
        print(f"{i:2d}. {formatear_regla(r)}")
else:
    max_soporte_relativo = max(r["soporte"] / total_filas for r in reglas_ordenadas)
    print(
        f"Ninguna regla alcanza un soporte >= {MIN_SOPORTE} sobre el total del "
        f"dataset (el soporte relativo máximo observado es "
        f"{max_soporte_relativo:.2f}, porque cada regla predice una sola "
        f"clase y las clases ocupan ~50% del dataset cada una)."
    )
    print(
        f"\nCon confianza >= {MIN_CONFIANZA} (sin exigir ese soporte), las "
        f"mejores reglas son:"
    )
    mejores_por_confianza = [
        r for r in reglas_ordenadas if r["precision"] >= MIN_CONFIANZA
    ]
    for i, r in enumerate(mejores_por_confianza[:10], 1):
        print(f"{i:2d}. {formatear_regla(r)}")

filas_export = []
for r in reglas_ordenadas:
    condicion_txt = " AND ".join(f"{a}={v}" for a, v in r["condiciones"])
    filas_export.append(
        {
            "regla_SI": condicion_txt,
            "clase_ENTONCES": r["clase"],
            "precision": r["precision"],
            "soporte": r["soporte"],
            "aciertos": r["aciertos"],
            "cobertura_clase": r["cobertura_clase"],
        }
    )

df_export = pd.DataFrame(filas_export)
df_export.to_csv("reglas_prism.csv", index=False, encoding="utf-8-sig")
print(f"\nSe exportaron {len(df_export)} reglas a 'reglas_prism.csv'")

