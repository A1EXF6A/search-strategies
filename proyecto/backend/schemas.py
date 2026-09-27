import math
from typing import Annotated

from pydantic import BaseModel, Field, StrictFloat, field_validator, model_validator

from config import INPUT_VARIABLES, VARIABLE_SPECS


class PredictionInput(BaseModel):
    air_temperature: Annotated[float, StrictFloat]
    process_temperature: Annotated[float, StrictFloat]
    rotational_speed: Annotated[float, StrictFloat]
    torque: Annotated[float, StrictFloat]
    tool_wear: Annotated[float, StrictFloat]
    use_optimized: bool = True

    @field_validator("air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear")
    @classmethod
    def validate_finite_and_range(cls, value: float, info) -> float:
        if not math.isfinite(value):
            raise ValueError("El valor debe ser un número finito")

        limits: dict[str, float] = INPUT_VARIABLES[info.field_name]

        if not (limits["min"] <= value <= limits["max"]):
            raise ValueError(
                f"Debe estar entre {limits['min']} y {limits['max']} "
                f"(rangos del dataset AI4I 2020)"
            )

        return value

    @model_validator(mode="after")
    def check_temperature_coherence(self) -> "PredictionInput":
        gap: float = self.process_temperature - self.air_temperature
        gap_spec: dict[str, float] = VARIABLE_SPECS["thermal_gap"]

        if gap <= gap_spec["min"]:
            raise ValueError(
                "La temperatura del proceso debe ser mayor que la temperatura del aire"
            )

        if gap > gap_spec["max"]:
            raise ValueError(
                f"La brecha térmica (proceso − aire) es {gap} K y supera el rango "
                f"modelado ({gap_spec['min']}–{gap_spec['max']} K) del dataset AI4I 2020. "
                "Ajusta las temperaturas para que la brecha quede dentro del rango."
            )

        return self


class RuleActivation(BaseModel):
    name: str
    cause: str
    strength: float = Field(ge=0.0, le=1.0)
    antecedents: list[tuple[str, str]] = Field(default_factory=list)
    consequent: str = ""
    consequent_display: str = ""


class PredictionResponse(BaseModel):
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: str
    risk_level_display: str
    action: str
    thermal_gap: float
    factors: list[str]
    activated_rules: list[RuleActivation]
    memberships: dict[str, dict[str, float]]
    output_strengths: dict[str, float]
    params_used: str = "optimized"