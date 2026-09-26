import json
from pathlib import Path
from typing import cast

from config import (
    DEFAULT_RULE_LABEL,
    FACTOR_DESCRIPTIONS,
    MAX_EXPLANATION_FACTORS,
    MISSING_PARAMS_MESSAGE,
    MODEL_VERSION,
    OPTIMIZED_PARAMS_PATH,
    RISK_LEVELS,
    RULE_LABELS,
)
from fuzzy import FuzzySystem
from schemas import PredictionInput, PredictionResponse, RuleActivation

DEFAULT_FACTOR: str = "Condiciones dentro de los rangos normales"


def load_parameters(path: Path = OPTIMIZED_PARAMS_PATH) -> dict[str, object]:
    if not path.exists():
        raise RuntimeError(MISSING_PARAMS_MESSAGE)

    with path.open("r", encoding="utf-8") as file:
        return cast(dict[str, object], json.load(file))


def risk_level_info(score: float) -> dict[str, str]:
    for entry in RISK_LEVELS:
        if score <= float(cast(float, entry["max"])):
            return {
                "level": str(entry["level"]),
                "level_display": str(entry["level_display"]),
                "action": str(entry["action"]),
            }

    return {
        "level": str(RISK_LEVELS[-1]["level"]),
        "level_display": str(RISK_LEVELS[-1]["level_display"]),
        "action": str(RISK_LEVELS[-1]["action"]),
    }


def build_factors(activated_rules: list[dict[str, object]]) -> list[str]:
    factors: list[str] = []

    ordered: list[dict[str, object]] = sorted(
        activated_rules, key=lambda rule: float(cast(float, rule["strength"])), reverse=True
    )

    for rule in ordered:
        antecedents = cast(list[tuple[str, str]], rule["antecedents"])

        for variable, term in antecedents:
            phrase: str | None = FACTOR_DESCRIPTIONS.get((variable, term))

            if phrase is not None and phrase not in factors:
                factors.append(phrase)

            if len(factors) >= MAX_EXPLANATION_FACTORS:
                break

        if len(factors) >= MAX_EXPLANATION_FACTORS:
            break

    return factors if factors else [DEFAULT_FACTOR]


class MaintenanceModel:
    def __init__(self, params_path: Path = OPTIMIZED_PARAMS_PATH) -> None:
        parameters: dict[str, object] = load_parameters(params_path)

        self.version: str = str(parameters.get("version", MODEL_VERSION))
        self.system: FuzzySystem = FuzzySystem(
            cast(dict[str, dict[str, float]], parameters["parameters"])
        )

    def predict(self, inputs: PredictionInput) -> PredictionResponse:
        thermal_gap: float = round(inputs.process_temperature - inputs.air_temperature, 2)

        row: dict[str, float] = {
            "process_temperature": inputs.process_temperature,
            "thermal_gap": thermal_gap,
            "rotational_speed": inputs.rotational_speed,
            "torque": inputs.torque,
            "tool_wear": inputs.tool_wear,
        }

        result: dict[str, object] = self.system.predict(row)

        risk_score: float = float(cast(float, result["risk_score"]))

        level: dict[str, str] = risk_level_info(risk_score)

        factors: list[str] = build_factors(
            cast(list[dict[str, object]], result["activated_rules"])
        )

        activated_rules: list[RuleActivation] = []

        for rule in cast(list[dict[str, object]], result["activated_rules"]):
            rule_name: str = str(rule["name"])

            activated_rules.append(
                RuleActivation(
                    name=rule_name,
                    cause=RULE_LABELS.get(rule_name, DEFAULT_RULE_LABEL),
                    strength=float(cast(float, rule["strength"])),
                    antecedents=cast(list[tuple[str, str]], rule["antecedents"]),
                    consequent=str(rule["consequent"]),
                    consequent_display=str(rule["consequent_display"]),
                )
            )

        memberships: dict[str, dict[str, float]] = cast(
            dict[str, dict[str, float]], result["memberships"]
        )

        return PredictionResponse(
            risk_score=risk_score,
            risk_level=level["level"],
            risk_level_display=level["level_display"],
            action=level["action"],
            thermal_gap=thermal_gap,
            factors=factors,
            activated_rules=activated_rules,
            memberships=memberships,
        )