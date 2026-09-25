import math
from typing import Annotated

from pydantic import BaseModel, Field, StrictFloat, field_validator, model_validator

from config import INPUT_VARIABLES


class PredictionInput(BaseModel):
    air_temperature: Annotated[float, StrictFloat]
    process_temperature: Annotated[float, StrictFloat]
    rotational_speed: Annotated[float, StrictFloat]
    torque: Annotated[float, StrictFloat]
    tool_wear: Annotated[float, StrictFloat]

    @field_validator("*")
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
        if self.process_temperature <= self.air_temperature:
            raise ValueError(
                "La temperatura del proceso debe ser mayor que la temperatura del aire"
            )

        return self


class RuleActivation(BaseModel):
    name: str
    cause: str
    strength: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: str
    risk_level_display: str
    action: str
    thermal_gap: float
    factors: list[str]
    activated_rules: list[RuleActivation]
    memberships: dict[str, dict[str, float]]