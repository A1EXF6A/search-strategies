import json
from datetime import datetime, timezone
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from config import (
    CROSSOVER_RATE,
    DATASET_COLUMNS,
    DATASET_ENCODING,
    DATASET_PATH,
    DEFAULT_PARAMETERS,
    ELITE_COUNT,
    FITNESS_SAMPLE_SIZE,
    FUZZY_VARIABLES,
    GENERATIONS,
    MODEL_VERSION,
    MUTATION_RATE,
    OPTIMIZED_PARAMS_PATH,
    POPULATION_SIZE,
    RANDOM_SEED,
    RISK_THRESHOLD,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from fuzzy import FloatArray, FuzzySystem
from genetic import ClassificationMetrics, GeneticOptimizer, evaluate_predictions


def load_dataset() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(DATASET_PATH, encoding=DATASET_ENCODING)

    rename_map: dict[str, str] = {}

    for name, column in DATASET_COLUMNS.items():
        rename_map[column] = name

    df = df.rename(columns=rename_map)

    df["thermal_gap"] = df["process_temperature"] - df["air_temperature"]

    return df


def make_values(df: pd.DataFrame) -> dict[str, FloatArray]:
    values: dict[str, FloatArray] = {}

    for variable in FUZZY_VARIABLES:
        values[variable] = df[variable].to_numpy(dtype=float)

    return values


SplitResult = tuple[pd.DataFrame, pd.DataFrame, NDArray[np.int64], NDArray[np.int64]]


def split_dataset(
    df: pd.DataFrame, y: NDArray[np.int64]
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]
]:
    first_split: SplitResult = cast(
        SplitResult,
        train_test_split(
            df, y, test_size=1.0 - TRAIN_FRACTION, stratify=y, random_state=RANDOM_SEED
        ),
    )

    train_df, temp_df, y_train, y_temp = first_split

    test_size: float = TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION)

    second_split: SplitResult = cast(
        SplitResult,
        train_test_split(
            temp_df, y_temp, test_size=test_size, stratify=y_temp, random_state=RANDOM_SEED
        ),
    )

    val_df, test_df, y_val, y_test = second_split

    return train_df, val_df, test_df, y_train, y_val, y_test


def print_metrics(label: str, metrics: ClassificationMetrics) -> None:
    matrix: list[list[int]] = metrics.confusion_matrix()

    print(f"  [{label}]")
    print(
        f"    balanced accuracy={metrics.balanced_accuracy:.4f}  "
        f"accuracy={metrics.accuracy:.4f}  "
        f"precision={metrics.precision:.4f}  "
        f"recall={metrics.recall:.4f}  f1={metrics.f1:.4f}"
    )
    print(
        f"    matriz de confusión: TP={matrix[0][0]}  FP={matrix[0][1]}  "
        f"FN={matrix[1][0]}  TN={matrix[1][1]}"
    )


def main() -> None:
    print("Optimización del sistema difuso con algoritmo genético")
    print("=" * 60)
    print()

    df: pd.DataFrame = load_dataset()

    y: NDArray[np.int64] = df["machine_failure"].to_numpy(dtype=np.int64)

    print(f"Dataset: {DATASET_PATH.name} ({len(df)} filas)")
    print(f"Fallo de máquina: {int(y.sum())} positivos")
    print()

    features: pd.DataFrame = cast(pd.DataFrame, df[list(FUZZY_VARIABLES)])

    train_df, val_df, test_df, y_train, y_val, y_test = split_dataset(features, y)

    print(
        f"División: entrenamiento {len(train_df)} ({len(train_df) / len(df):.0%}), "
        f"validación {len(val_df)} ({len(val_df) / len(df):.0%}), "
        f"test {len(test_df)} ({len(test_df) / len(df):.0%})"
    )
    print()

    values_train: dict[str, FloatArray] = make_values(train_df)
    values_val: dict[str, FloatArray] = make_values(val_df)
    values_test: dict[str, FloatArray] = make_values(test_df)

    print("Sistema difuso sin optimización (parámetros por defecto):")
    print("-" * 60)

    baseline_system: FuzzySystem = FuzzySystem(DEFAULT_PARAMETERS)

    risk_baseline_train, _, _ = baseline_system.evaluate_batch(values_train)
    risk_baseline_test, _, _ = baseline_system.evaluate_batch(values_test)

    baseline_train_metrics: ClassificationMetrics = evaluate_predictions(
        y_train, (risk_baseline_train >= RISK_THRESHOLD).astype(np.int64)
    )
    baseline_test_metrics: ClassificationMetrics = evaluate_predictions(
        y_test, (risk_baseline_test >= RISK_THRESHOLD).astype(np.int64)
    )

    print_metrics("Entrenamiento", baseline_train_metrics)
    print_metrics("Test", baseline_test_metrics)
    print()

    print("Algoritmo genético (DEAP):")
    print("-" * 60)
    print(
        f"  población={POPULATION_SIZE}, generaciones={GENERATIONS}, "
        f"élite={ELITE_COUNT}, crossover={CROSSOVER_RATE}, mutación={MUTATION_RATE}, "
        f"fitness: balanced accuracy sobre muestra estratificada de "
        f"{FITNESS_SAMPLE_SIZE} filas, seed={RANDOM_SEED}"
    )
    print()

    optimizer: GeneticOptimizer = GeneticOptimizer(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        elite_count=ELITE_COUNT,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        seed=RANDOM_SEED,
    )

    result = optimizer.run(values_train, y_train, values_val, y_val)

    best_parameters: dict[str, dict[str, float]] = result.best_parameters

    print("Sistema optimizado por GA:")
    print("-" * 60)

    assert result.validation_metrics is not None

    print_metrics("Validación", result.validation_metrics)

    optimized_system: FuzzySystem = FuzzySystem(best_parameters)

    risk_optimized_train, _, _ = optimized_system.evaluate_batch(values_train)
    risk_optimized_test, _, _ = optimized_system.evaluate_batch(values_test)

    optimized_train_metrics: ClassificationMetrics = evaluate_predictions(
        y_train, (risk_optimized_train >= RISK_THRESHOLD).astype(np.int64)
    )
    optimized_test_metrics: ClassificationMetrics = evaluate_predictions(
        y_test, (risk_optimized_test >= RISK_THRESHOLD).astype(np.int64)
    )

    print_metrics("Entrenamiento", optimized_train_metrics)
    print_metrics("Test", optimized_test_metrics)
    print()

    print("Comparación antes / después (mismo conjunto de test):")
    print("-" * 60)

    delta_ba: float = optimized_test_metrics.balanced_accuracy - baseline_test_metrics.balanced_accuracy
    delta_precision: float = optimized_test_metrics.precision - baseline_test_metrics.precision
    delta_recall: float = optimized_test_metrics.recall - baseline_test_metrics.recall
    delta_f1: float = optimized_test_metrics.f1 - baseline_test_metrics.f1

    print(f"  Balanced Accuracy: {baseline_test_metrics.balanced_accuracy:.3f} -> "
          f"{optimized_test_metrics.balanced_accuracy:.3f}  ({delta_ba:+.3f})")
    print(f"  Precision:         {baseline_test_metrics.precision:.3f} -> "
          f"{optimized_test_metrics.precision:.3f}  ({delta_precision:+.3f})")
    print(f"  Recall:            {baseline_test_metrics.recall:.3f} -> "
          f"{optimized_test_metrics.recall:.3f}  ({delta_recall:+.3f})")
    print(f"  F1:                {baseline_test_metrics.f1:.3f} -> "
          f"{optimized_test_metrics.f1:.3f}  ({delta_f1:+.3f})")
    print()

    print(f"Mejor fitness durante la evolución: {result.best_fitness:.4f} "
          f"(balanced accuracy sobre muestra de fitness)")
    print()

    OPTIMIZED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "version": MODEL_VERSION,
        "seed": RANDOM_SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hyperparameters": {
            "population_size": POPULATION_SIZE,
            "generations": GENERATIONS,
            "elite_count": ELITE_COUNT,
            "crossover_rate": CROSSOVER_RATE,
            "mutation_rate": MUTATION_RATE,
            "risk_threshold": RISK_THRESHOLD,
            "fitness_sample_size": FITNESS_SAMPLE_SIZE,
        },
        "fitness": {
            "balanced_accuracy_best_generation": round(result.best_fitness, 4),
            "balanced_accuracy_train": round(optimized_train_metrics.balanced_accuracy, 4),
            "validation": result.validation_metrics.to_dict(),
        },
        "parameters": best_parameters,
        "baseline_metrics": baseline_test_metrics.to_dict(),
        "optimized_metrics": optimized_test_metrics.to_dict(),
    }

    with OPTIMIZED_PARAMS_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(f"Parámetros guardados en: {OPTIMIZED_PARAMS_PATH}")
    print("Optimización completada.")


if __name__ == "__main__":
    main()