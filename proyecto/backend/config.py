import json
from pathlib import Path
from typing import TypedDict, cast

BACKEND_DIR: Path = Path(__file__).parent

DATASET_PATH: Path = BACKEND_DIR / "data" / "ai4i2020.csv"

ARTIFACTS_DIR: Path = BACKEND_DIR / "artifacts"

OPTIMIZED_PARAMS_PATH: Path = ARTIFACTS_DIR / "optimized_parameters.json"

MINED_RULES_PATH: Path = ARTIFACTS_DIR / "mined_rules.json"

DATASET_ENCODING: str = "utf-8-sig"

DATASET_COLUMNS: dict[str, str] = {
    "air_temperature": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
    "machine_failure": "Machine failure",
}

INPUT_VARIABLES: dict[str, dict[str, float]] = {
    "air_temperature": {"min": 290.0, "max": 310.0},
    "process_temperature": {"min": 300.0, "max": 320.0},
    "rotational_speed": {"min": 1100.0, "max": 3000.0},
    "torque": {"min": 0.0, "max": 90.0},
    "tool_wear": {"min": 0.0, "max": 260.0},
}

FUZZY_VARIABLES: tuple[str, ...] = (
    "process_temperature",
    "thermal_gap",
    "rotational_speed",
    "torque",
    "tool_wear",
)

TERMS: tuple[str, ...] = ("low", "medium", "high")


class VariableSpec(TypedDict):
    display: str
    unit: str
    min: float
    max: float


VARIABLE_SPECS: dict[str, VariableSpec] = {
    "air_temperature": {"display": "Temperatura del aire", "unit": "K", "min": 295.0, "max": 310.0},
    "process_temperature": {"display": "Temperatura del proceso", "unit": "K", "min": 300.0, "max": 320.0},
    "rotational_speed": {"display": "Velocidad de rotación", "unit": "rpm", "min": 1100.0, "max": 3000.0},
    "torque": {"display": "Torque", "unit": "Nm", "min": 0.0, "max": 90.0},
    "tool_wear": {"display": "Desgaste de herramienta", "unit": "min", "min": 0.0, "max": 260.0},
    "thermal_gap": {"display": "Brecha térmica", "unit": "K", "min": 0.0, "max": 15.0},
}

DEFAULT_PARAMETERS: dict[str, dict[str, float]] = {
    "process_temperature": {"p1": 306.7, "p2": 310.0, "p3": 313.3},
    "thermal_gap": {"p1": 5.0, "p2": 7.5, "p3": 10.0},
    "rotational_speed": {"p1": 1733.3, "p2": 2050.0, "p3": 2366.7},
    "torque": {"p1": 30.0, "p2": 45.0, "p3": 60.0},
    "tool_wear": {"p1": 86.7, "p2": 130.0, "p3": 173.3},
}

RISK_MIN: float = 0.0
RISK_MAX: float = 100.0

OUTPUT_TERMS: tuple[str, ...] = ("low", "medium", "high", "critical")

OUTPUT_MEMBERSHIPS: dict[str, tuple[float, ...]] = {
    "low": (0.0, 0.0, 25.0, 45.0),
    "medium": (25.0, 45.0, 65.0),
    "high": (50.0, 70.0, 90.0),
    "critical": (75.0, 90.0, 100.0, 100.0),
}

UNIVERSE_POINTS: int = 41

RISK_LEVELS: tuple[dict[str, object], ...] = (
    {"max": 24, "level": "bajo", "level_display": "BAJO", "action": "Continuar operación"},
    {"max": 49, "level": "moderado", "level_display": "MODERADO", "action": "Monitorear"},
    {"max": 74, "level": "alto", "level_display": "ALTO", "action": "Programar mantenimiento"},
    {"max": 100, "level": "critico", "level_display": "CRÍTICO", "action": "Detener y revisar"},
)

RISK_THRESHOLD: float = 50.0

def load_fuzzy_rules() -> tuple[dict[str, object], ...]:
    rules: list[dict[str, object]] = []

    if not MINED_RULES_PATH.exists():
        return tuple(rules)

    with MINED_RULES_PATH.open("r", encoding="utf-8") as file:
        data: dict[str, object] = cast(dict[str, object], json.load(file))

    for entry in cast(list[dict[str, object]], data["rules"]):
        raw_antecedents: list[list[str]] = cast(
            list[list[str]], entry["antecedents"]
        )
        antecedents: list[tuple[str, str]] = []

        for variable, term in raw_antecedents:
            antecedents.append((str(variable), str(term)))

        rules.append(
            {
                "name": str(entry["name"]),
                "antecedents": tuple(antecedents),
                "consequent": str(entry["consequent"]),
            }
        )

    return tuple(rules)


FUZZY_RULES: tuple[dict[str, object], ...] = load_fuzzy_rules()

MISSING_RULES_MESSAGE: str = (
    "No existe mined_rules.json. "
    "Ejecute el proceso de minería de reglas antes de iniciar el sistema."
)

FACTOR_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("process_temperature", "low"): "Temperatura de proceso baja",
    ("process_temperature", "medium"): "Temperatura de proceso moderada",
    ("process_temperature", "high"): "Temperatura de proceso elevada",
    ("thermal_gap", "low"): "Condición térmica desfavorable (brecha térmica baja)",
    ("thermal_gap", "medium"): "Brecha térmica moderada",
    ("thermal_gap", "high"): "Brecha térmica elevada",
    ("rotational_speed", "low"): "Velocidad de rotación baja",
    ("rotational_speed", "medium"): "Velocidad de rotación moderada",
    ("rotational_speed", "high"): "Velocidad de rotación elevada",
    ("torque", "low"): "Torque bajo",
    ("torque", "medium"): "Torque moderado",
    ("torque", "high"): "Torque elevado",
    ("tool_wear", "low"): "Desgaste de herramienta bajo",
    ("tool_wear", "medium"): "Desgaste de herramienta moderado",
    ("tool_wear", "high"): "Desgaste de herramienta elevado",
}

MAX_EXPLANATION_FACTORS: int = 4

RULE_LABELS: dict[str, str] = {
    "PR1_torque_high_thermal_gap_medium_tool_wear_low_process_temperature_high_rotational_speed_low": (
        "Sobrecarga de potencia — torque alto, rotación baja, brecha térmica moderada, "
        "temperatura de proceso alta"
    ),
    "PR2_torque_high_thermal_gap_medium_tool_wear_low_process_temperature_medium_rotational_speed_low": (
        "Disipación térmica comprometida — torque alto, rotación baja, brecha térmica moderada, "
        "desgaste bajo"
    ),
    "PR3_torque_high_thermal_gap_medium_tool_wear_medium_process_temperature_medium_rotational_speed_low": (
        "Disipación térmica comprometida — torque alto, rotación baja, brecha térmica moderada, "
        "desgaste medio"
    ),
    "PR4_torque_high_thermal_gap_medium_process_temperature_medium_rotational_speed_low_tool_wear_high": (
        "Sobrecalentamiento con desgaste avanzado — torque alto, rotación baja, "
        "brecha térmica moderada, herramienta muy desgastada"
    ),
    "PR5_rotational_speed_high_tool_wear_low_process_temperature_high_thermal_gap_high_torque_low": (
        "Condición anómala mixta — velocidad elevada, torque bajo, desgaste bajo"
    ),
    "PR6_torque_high_tool_wear_high_process_temperature_high_thermal_gap_high_rotational_speed_low": (
        "Desgaste avanzado bajo sobreesfuerzo — herramienta muy desgastada, torque alto, rotación baja"
    ),
    "PR7_torque_high_tool_wear_high_process_temperature_low_thermal_gap_high_rotational_speed_low": (
        "Sobreesfuerzo mecánico — torque alto, rotación baja, herramienta muy desgastada"
    ),
    "PS1_torque_low_rotational_speed_low_tool_wear_low": (
        "Operación segura — torque bajo, rotación baja, desgaste bajo"
    ),
    "PS2_torque_medium_thermal_gap_high_tool_wear_medium": (
        "Operación segura — torque medio, brecha térmica elevada, desgaste medio"
    ),
    "PS3_rotational_speed_medium_tool_wear_medium": (
        "Operación segura — rotación media, desgaste medio"
    ),
    "PS4_rotational_speed_medium_thermal_gap_medium": (
        "Operación segura — rotación media, brecha térmica media"
    ),
}

DEFAULT_RULE_LABEL: str = "Regla sin etiqueta específica"

POPULATION_SIZE: int = 30
GENERATIONS: int = 50
ELITE_COUNT: int = 2
CROSSOVER_RATE: float = 0.8
MUTATION_RATE: float = 0.1
GENES_PER_VARIABLE: int = 3
RANDOM_SEED: int = 42
MIN_SEPARATION_FRACTION: float = 0.02
FITNESS_SAMPLE_SIZE: int = 600

TRAIN_FRACTION: float = 0.70
VALIDATION_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15

MODEL_VERSION: str = "1.0"

MISSING_PARAMS_MESSAGE: str = (
    "No existen parámetros optimizados. "
    "Ejecute el proceso de entrenamiento antes de iniciar el sistema."
)