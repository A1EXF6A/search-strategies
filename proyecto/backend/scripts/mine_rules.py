import json
from datetime import datetime, timezone
from typing import cast

import pandas as pd

from config import (
    DATASET_COLUMNS,
    DATASET_ENCODING,
    DATASET_PATH,
    DEFAULT_PARAMETERS,
    MINED_RULES_PATH,
    MODEL_VERSION,
    TERMS,
    UNIVERSE_POINTS,
)
from prism import (
    MAX_FAILURE_RULES,
    MAX_SAFE_RULES,
    MIN_CONFIDENCE_FAILURE,
    MIN_CONFIDENCE_SAFE,
    MIN_COVERAGE_FAILURE,
    MIN_COVERAGE_SAFE,
    MinedRule,
    build_fuzzy_rules,
    discretize,
    run_prism,
)


def load_dataset() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(DATASET_PATH, encoding=DATASET_ENCODING)

    rename_map: dict[str, str] = {}

    for name, column in DATASET_COLUMNS.items():
        rename_map[column] = name

    df = df.rename(columns=rename_map)

    df["thermal_gap"] = df["process_temperature"] - df["air_temperature"]

    return df


def print_rules(label: str, rules: list[MinedRule]) -> None:
    print(f"{label}: {len(rules)} reglas")
    print("-" * 70)

    for index, rule in enumerate(rules, 1):
        conditions: list[tuple[str, str]] = cast(
            list[tuple[str, str]], rule["conditions"]
        )
        parts: list[str] = []

        for variable, term in conditions:
            parts.append(f"{variable} = {term}")

        antecedent: str = " AND ".join(parts)

        print(f"  {index:>2}. [{antecedent}]")
        print(
            f"      confianza={float(cast(float, rule['confidence'])) * 100:.1f}%  "
            f"cobertura={rule['coverage']}  lift={float(cast(float, rule['lift'])):.1f}"
        )

    print()


def main() -> None:
    print("Minería de reglas difusas con PRISM")
    print("=" * 70)
    print()

    df: pd.DataFrame = load_dataset()

    y: pd.Series = cast(pd.Series, df["machine_failure"].astype(int))

    print(f"Dataset: {DATASET_PATH.name} ({len(df)} filas, {int(y.sum())} fallos)")
    print()

    discretized: pd.DataFrame = discretize(df, DEFAULT_PARAMETERS)
    discretized["failure"] = y.astype(str)
    discretized["safe"] = (1 - y).astype(str)

    failure_rules: list[MinedRule] = run_prism(
        discretized,
        "failure",
        MIN_CONFIDENCE_FAILURE,
        MIN_COVERAGE_FAILURE,
        MAX_FAILURE_RULES,
    )

    safe_rules: list[MinedRule] = run_prism(
        discretized,
        "safe",
        MIN_CONFIDENCE_SAFE,
        MIN_COVERAGE_SAFE,
        MAX_SAFE_RULES,
    )

    print("Reglas descubiertas por PRISM sobre el dataset discretizado:")
    print()
    print_rules("Fallo de máquina", failure_rules)
    print_rules("Sin fallo (seguras)", safe_rules)

    fuzzy_rules: tuple[dict[str, object], ...] = build_fuzzy_rules(
        failure_rules, safe_rules
    )

    print(f"Base difusa generada: {len(fuzzy_rules)} reglas")
    print()

    MINED_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "PRISM",
        "dataset": DATASET_PATH.name,
        "discretization": {
            "parameters": DEFAULT_PARAMETERS,
            "universe_points": UNIVERSE_POINTS,
            "terms": list(TERMS),
        },
        "mining": {
            "min_confidence_failure": MIN_CONFIDENCE_FAILURE,
            "min_coverage_failure": MIN_COVERAGE_FAILURE,
            "min_confidence_safe": MIN_CONFIDENCE_SAFE,
            "min_coverage_safe": MIN_COVERAGE_SAFE,
        },
        "rules": list(fuzzy_rules),
    }

    with MINED_RULES_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(f"Reglas guardadas en: {MINED_RULES_PATH}")
    print("Minería completada.")


if __name__ == "__main__":
    main()