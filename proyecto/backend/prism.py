from typing import cast

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from numpy.typing import NDArray

from config import FUZZY_VARIABLES, TERMS, UNIVERSE_POINTS, VARIABLE_SPECS, VariableSpec

FloatArray = NDArray[np.float64]

Condition = tuple[str, str]

MIN_CONFIDENCE_FAILURE: float = 0.40
MIN_COVERAGE_FAILURE: int = 5
MIN_CONFIDENCE_SAFE: float = 0.965
MIN_COVERAGE_SAFE: int = 50
MAX_FAILURE_RULES: int = 20
MAX_SAFE_RULES: int = 4

MinedRule = dict[str, object]


def build_universe(spec: VariableSpec) -> FloatArray:
    return np.linspace(spec["min"], spec["max"], UNIVERSE_POINTS)


def membership_arrays(
    spec: VariableSpec, parameters: dict[str, float]
) -> dict[str, FloatArray]:
    universe: FloatArray = build_universe(spec)
    p1: float = parameters["p1"]
    p2: float = parameters["p2"]
    p3: float = parameters["p3"]

    return {
        "low": fuzz.trapmf(universe, [spec["min"], spec["min"], p1, p2]),
        "medium": fuzz.trimf(universe, [p1, p2, p3]),
        "high": fuzz.trapmf(universe, [p2, p3, spec["max"], spec["max"]]),
    }


def discretize(
    df: pd.DataFrame, parameters: dict[str, dict[str, float]]
) -> pd.DataFrame:
    discretized: pd.DataFrame = pd.DataFrame(index=df.index)

    for variable in FUZZY_VARIABLES:
        spec: VariableSpec = VARIABLE_SPECS[variable]
        universe: FloatArray = build_universe(spec)
        arrays: dict[str, FloatArray] = membership_arrays(spec, parameters[variable])
        terms: list[str] = []

        for value in df[variable].to_numpy(dtype=float):
            index: int = int(np.argmin(np.abs(universe - value)))
            best_term: str = TERMS[0]
            best_score: float = -1.0

            for term in TERMS:
                score: float = float(arrays[term][index])

                if score > best_score:
                    best_score = score
                    best_term = term

            terms.append(best_term)

        discretized[variable] = terms

    return discretized


def count_matches(df: pd.DataFrame, conditions: list[Condition]) -> pd.Series:
    mask: pd.Series = pd.Series(True, index=df.index)

    for variable, term in conditions:
        mask &= df[variable] == term

    return mask


def best_condition(
    df: pd.DataFrame, target: str, used: list[Condition]
) -> tuple[Condition, float, int] | None:
    used_variables: set[str] = set()

    for variable, _ in used:
        used_variables.add(variable)

    best: tuple[Condition, float, int] | None = None

    for variable in FUZZY_VARIABLES:
        if variable in used_variables:
            continue

        for term in df[variable].unique():
            mask: pd.Series = df[variable] == term
            total: int = int(mask.sum())

            if total == 0:
                continue

            positives: int = int((mask & (df[target] == "1")).sum())
            confidence: float = positives / total

            if (
                best is None
                or confidence > best[1]
                or (confidence == best[1] and total > best[2])
            ):
                best = ((variable, str(term)), confidence, total)

    return best


def run_prism(
    df: pd.DataFrame,
    target: str,
    min_confidence: float,
    min_coverage: int,
    max_rules: int,
) -> list[MinedRule]:
    remaining: pd.DataFrame = df
    rules: list[MinedRule] = []
    target_count: int = int((df[target] == "1").sum())
    total_rows: int = df.shape[0]

    while int((remaining[target] == "1").sum()) > 0 and len(rules) < max_rules:
        conditions: list[Condition] = []
        working: pd.DataFrame = remaining

        while True:
            choice: tuple[Condition, float, int] | None = best_condition(
                working, target, conditions
            )

            if choice is None:
                break

            conditions.append(choice[0])
            working = cast(pd.DataFrame, working[count_matches(working, conditions)])

            if bool(working[target].eq("1").all()):
                break

        if not conditions:
            break

        mask: pd.Series = count_matches(df, conditions)
        coverage: int = int(mask.sum())
        positives: int = int((mask & (df[target] == "1")).sum())
        confidence: float = positives / coverage if coverage > 0 else 0.0

        if confidence >= min_confidence and coverage >= min_coverage:
            rules.append(
                {
                    "conditions": conditions,
                    "coverage": coverage,
                    "confidence": confidence,
                    "lift": confidence / (target_count / total_rows),
                }
            )

        remaining = cast(
            pd.DataFrame, remaining[~count_matches(remaining, conditions)]
        )

    return rules


def rule_name(prefix: str, index: int, conditions: list[Condition]) -> str:
    parts: list[str] = []

    for variable, term in conditions:
        parts.append(f"{variable}_{term}")

    return f"{prefix}{index}_{'_'.join(parts)}"


def build_fuzzy_rules(
    failure_rules: list[MinedRule], safe_rules: list[MinedRule]
) -> tuple[dict[str, object], ...]:
    generated: list[dict[str, object]] = []

    ordered_failure: list[MinedRule] = sorted(
        failure_rules, key=lambda rule: -float(cast(float, rule["confidence"]))
    )

    for index, rule in enumerate(ordered_failure, 1):
        conditions: list[Condition] = cast(list[Condition], rule["conditions"])
        confidence: float = float(cast(float, rule["confidence"]))
        consequent: str = "critical" if confidence >= 0.75 else "high"

        generated.append(
            {
                "name": rule_name("PR", index, conditions),
                "antecedents": tuple(conditions),
                "consequent": consequent,
                "confidence": confidence,
                "coverage": int(cast(int, rule["coverage"])),
                "lift": float(cast(float, rule["lift"])),
                "class": "failure",
            }
        )

    ordered_safe: list[MinedRule] = sorted(
        safe_rules, key=lambda rule: -int(cast(int, rule["coverage"]))
    )

    for index, rule in enumerate(ordered_safe, 1):
        conditions = cast(list[Condition], rule["conditions"])

        generated.append(
            {
                "name": rule_name("PS", index, conditions),
                "antecedents": tuple(conditions),
                "consequent": "low",
                "confidence": float(cast(float, rule["confidence"])),
                "coverage": int(cast(int, rule["coverage"])),
                "lift": float(cast(float, rule["lift"])),
                "class": "safe",
            }
        )

    return tuple(generated)