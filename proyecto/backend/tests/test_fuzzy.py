import json
import numpy as np
import pandas as pd
import pytest
from collections.abc import Generator
from fastapi.testclient import TestClient
from typing import cast

from config import (
    DEFAULT_PARAMETERS,
    FUZZY_RULES,
    FUZZY_VARIABLES,
    MINED_RULES_PATH,
    OPTIMIZED_PARAMS_PATH,
    RISK_MAX,
    RISK_MIN,
)
from fuzzy import FuzzySystem
from genetic import GeneticOptimizer, decode_genes
from main import app
from prism import discretize, run_prism


def mid_row() -> dict[str, np.ndarray]:
    return {
        "process_temperature": np.array([310.0]),
        "thermal_gap": np.array([7.5]),
        "rotational_speed": np.array([2000.0]),
        "torque": np.array([45.0]),
        "tool_wear": np.array([130.0]),
    }


def row_values(row: dict[str, np.ndarray]) -> dict[str, float]:
    values: dict[str, float] = {}

    for variable in FUZZY_VARIABLES:
        values[variable] = float(row[variable][0])

    return values


def test_membership_medium_peak() -> None:
    system: FuzzySystem = FuzzySystem(DEFAULT_PARAMETERS)

    row: dict[str, np.ndarray] = mid_row()

    _, rule_strengths, _ = system.evaluate_batch(row)

    memberships: dict[str, dict[str, float]] = cast(
        dict[str, dict[str, float]],
        system.predict(row_values(row))[
            "memberships"
        ],
    )

    assert memberships["torque"]["medium"] == pytest.approx(1.0, abs=1e-3)
    assert memberships["torque"]["low"] == pytest.approx(0.0, abs=1e-3)
    assert rule_strengths.shape[1] == len(FUZZY_RULES)


def test_membership_low_and_high_plateaus() -> None:
    system: FuzzySystem = FuzzySystem(DEFAULT_PARAMETERS)

    row: dict[str, np.ndarray] = mid_row()
    row["tool_wear"] = np.array([2.0])

    memberships: dict[str, dict[str, float]] = cast(
        dict[str, dict[str, float]],
        system.predict(row_values(row))[
            "memberships"
        ],
    )

    assert memberships["tool_wear"]["low"] == pytest.approx(1.0, abs=1e-3)

    row["tool_wear"] = np.array([250.0])

    memberships = cast(
        dict[str, dict[str, float]],
        system.predict(row_values(row))[
            "memberships"
        ],
    )

    assert memberships["tool_wear"]["high"] == pytest.approx(1.0, abs=1e-3)


def test_fuzzification_memberships_in_range() -> None:
    system: FuzzySystem = FuzzySystem(DEFAULT_PARAMETERS)

    row: dict[str, np.ndarray] = mid_row()

    memberships: dict[str, dict[str, float]] = cast(
        dict[str, dict[str, float]],
        system.predict(row_values(row))[
            "memberships"
        ],
    )

    for variable in FUZZY_VARIABLES:
        for term in ("low", "medium", "high"):
            value: float = float(memberships[variable][term])

            assert 0.0 <= value <= 1.0


def test_first_failure_rule_fires() -> None:
    system: FuzzySystem = FuzzySystem(DEFAULT_PARAMETERS)

    row: dict[str, np.ndarray] = {
        "process_temperature": np.array([312.0]),
        "thermal_gap": np.array([8.0]),
        "rotational_speed": np.array([1400.0]),
        "torque": np.array([70.0]),
        "tool_wear": np.array([20.0]),
    }

    _, rule_strengths, _ = system.evaluate_batch(row)

    assert rule_strengths[0, 0] > 0.5


def test_risk_score_within_range() -> None:
    system: FuzzySystem = FuzzySystem(DEFAULT_PARAMETERS)

    healthy: dict[str, np.ndarray] = {
        "process_temperature": np.array([306.0]),
        "thermal_gap": np.array([11.0]),
        "rotational_speed": np.array([1300.0]),
        "torque": np.array([8.0]),
        "tool_wear": np.array([2.0]),
    }

    dangerous: dict[str, np.ndarray] = {
        "process_temperature": np.array([310.0]),
        "thermal_gap": np.array([8.0]),
        "rotational_speed": np.array([1400.0]),
        "torque": np.array([70.0]),
        "tool_wear": np.array([20.0]),
    }

    for row in (healthy, dangerous):
        risk, _, _ = system.evaluate_batch(row)

        assert RISK_MIN <= float(risk[0]) <= RISK_MAX


def test_risk_higher_in_dangerous_case() -> None:
    system: FuzzySystem = FuzzySystem(DEFAULT_PARAMETERS)

    healthy: dict[str, np.ndarray] = {
        "process_temperature": np.array([306.0]),
        "thermal_gap": np.array([11.0]),
        "rotational_speed": np.array([1300.0]),
        "torque": np.array([8.0]),
        "tool_wear": np.array([2.0]),
    }

    dangerous: dict[str, np.ndarray] = {
        "process_temperature": np.array([310.0]),
        "thermal_gap": np.array([8.0]),
        "rotational_speed": np.array([1400.0]),
        "torque": np.array([70.0]),
        "tool_wear": np.array([20.0]),
    }

    risk_healthy, _, _ = system.evaluate_batch(healthy)
    risk_dangerous, _, _ = system.evaluate_batch(dangerous)

    assert float(risk_dangerous[0]) > float(risk_healthy[0])


def test_decode_genes_enforces_ordering() -> None:
    genes: np.ndarray = np.array(
        [0.9, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8, 0.3, 0.1, 0.7, 0.9, 0.2, 0.5, 0.6, 0.4]
    )

    parameters: dict[str, dict[str, float]] = decode_genes(genes)

    for variable in FUZZY_VARIABLES:
        cut: dict[str, float] = parameters[variable]

        assert cut["p1"] < cut["p2"] < cut["p3"]


def test_genetic_deterministic_with_seed() -> None:
    rng: np.random.Generator = np.random.default_rng(0)

    n: int = 200

    values: dict[str, np.ndarray] = {
        "process_temperature": np.linspace(305.0, 314.0, n),
        "thermal_gap": np.linspace(7.0, 12.0, n),
        "rotational_speed": np.linspace(1100.0, 2900.0, n),
        "torque": np.linspace(3.0, 80.0, n),
        "tool_wear": np.linspace(0.0, 253.0, n),
    }

    y: np.ndarray = rng.integers(0, 2, n).astype(np.int64)

    first: GeneticOptimizer = GeneticOptimizer(
        population_size=10, generations=5, seed=42
    )
    second: GeneticOptimizer = GeneticOptimizer(
        population_size=10, generations=5, seed=42
    )

    result_a = first.run(values, y)
    result_b = second.run(values, y)

    assert result_a.best_fitness == result_b.best_fitness
    assert result_a.best_parameters == result_b.best_parameters


@pytest.mark.skipif(
    not MINED_RULES_PATH.exists(), reason="requiere mined_rules.json"
)
def test_mined_rules_loaded_from_artifact() -> None:
    assert len(FUZZY_RULES) > 0

    with MINED_RULES_PATH.open("r", encoding="utf-8") as file:
        data: dict[str, object] = cast(dict[str, object], json.load(file))

    artifact_rules: list[dict[str, object]] = cast(
        list[dict[str, object]], data["rules"]
    )

    assert len(FUZZY_RULES) == len(artifact_rules)

    for entry in FUZZY_RULES:
        assert str(entry["name"]).startswith(("PR", "PS"))
        assert len(cast(tuple[tuple[str, str], ...], entry["antecedents"])) >= 1
        assert str(entry["consequent"]) in ("low", "high", "critical")


def test_discretize_maps_terms() -> None:
    df: pd.DataFrame = pd.DataFrame(
        {
            "process_temperature": [310.0],
            "thermal_gap": [7.5],
            "rotational_speed": [2000.0],
            "torque": [45.0],
            "tool_wear": [130.0],
        }
    )

    result: pd.DataFrame = discretize(df, DEFAULT_PARAMETERS)

    assert result["torque"].iloc[0] == "medium"
    assert result["thermal_gap"].iloc[0] == "medium"
    assert result["tool_wear"].iloc[0] == "medium"


def test_prism_synthetic_rules() -> None:
    data: dict[str, list[str]] = {
        "process_temperature": ["medium"] * 4,
        "thermal_gap": ["medium"] * 4,
        "rotational_speed": ["low", "low", "high", "high"],
        "torque": ["high", "high", "low", "low"],
        "tool_wear": ["low"] * 4,
        "failure": ["1", "1", "0", "0"],
    }

    df: pd.DataFrame = pd.DataFrame(data)

    rules: list[dict[str, object]] = run_prism(df, "failure", 0.5, 1, 10)

    assert len(rules) == 1

    conditions: list[tuple[str, str]] = cast(
        list[tuple[str, str]], rules[0]["conditions"]
    )

    assert conditions == [("rotational_speed", "low")]
    assert float(cast(float, rules[0]["confidence"])) == pytest.approx(1.0)
    assert int(cast(int, rules[0]["coverage"])) == 2


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skipif(
    not OPTIMIZED_PARAMS_PATH.exists(), reason="requiere optimized_parameters.json"
)
def test_predict_valid(client: TestClient) -> None:
    payload: dict[str, float | int] = {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100,
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 200

    body: dict = response.json()

    assert 0.0 <= body["risk_score"] <= 100.0
    assert body["risk_level"] in ("bajo", "moderado", "alto", "critico")
    assert body["thermal_gap"] == pytest.approx(10.0)
    assert isinstance(body["factors"], list)
    assert body["memberships"]["torque"]["high"] >= 0.0


def test_predict_missing_field(client: TestClient) -> None:
    payload: dict[str, float] = {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_predict_string_value(client: TestClient) -> None:
    payload: dict[str, object] = {
        "air_temperature": "caliente",
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100,
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_predict_out_of_range(client: TestClient) -> None:
    payload: dict[str, float] = {
        "air_temperature": 500.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100,
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_predict_incoherent_temperatures(client: TestClient) -> None:
    payload: dict[str, float] = {
        "air_temperature": 305.0,
        "process_temperature": 300.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100,
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422