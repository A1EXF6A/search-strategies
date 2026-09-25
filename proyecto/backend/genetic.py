import random
import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import numpy as np
from deap import base, creator, tools
from numpy.typing import NDArray

from config import (
    CROSSOVER_RATE,
    DEFAULT_PARAMETERS,
    ELITE_COUNT,
    FITNESS_SAMPLE_SIZE,
    FUZZY_VARIABLES,
    GENERATIONS,
    GENES_PER_VARIABLE,
    MIN_SEPARATION_FRACTION,
    MUTATION_RATE,
    POPULATION_SIZE,
    RANDOM_SEED,
    RISK_THRESHOLD,
    VARIABLE_SPECS,
    VariableSpec,
)
from fuzzy import FloatArray, FuzzySystem

GENE_RANGE: tuple[float, float] = (0.0, 1.0)
N_GENES: int = len(FUZZY_VARIABLES) * GENES_PER_VARIABLE
TOURNAMENT_SIZE: int = 3

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=getattr(creator, "FitnessMax"))

Individual = cast(type[list], getattr(creator, "Individual"))


@dataclass(frozen=True)
class ClassificationMetrics:
    balanced_accuracy: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    def confusion_matrix(self) -> list[list[int]]:
        return [
            [self.true_positives, self.false_positives],
            [self.false_negatives, self.true_negatives],
        ]

    def to_dict(self) -> dict[str, float | list[list[int]]]:
        return {
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix(),
        }


def evaluate_predictions(
    y_true: NDArray[np.int64], y_pred: NDArray[np.int64]
) -> ClassificationMetrics:
    tp: int = int(((y_pred == 1) & (y_true == 1)).sum())
    fp: int = int(((y_pred == 1) & (y_true == 0)).sum())
    tn: int = int(((y_pred == 0) & (y_true == 0)).sum())
    fn: int = int(((y_pred == 0) & (y_true == 1)).sum())

    total: int = tp + fp + tn + fn
    accuracy: float = (tp + tn) / total if total else 0.0
    precision: float = tp / (tp + fp) if (tp + fp) else 0.0
    recall: float = tp / (tp + fn) if (tp + fn) else 0.0
    specificity: float = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy: float = (recall + specificity) / 2.0
    f1: float = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )

    return ClassificationMetrics(
        balanced_accuracy=round(balanced_accuracy, 4),
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )


def decode_genes(genes: FloatArray) -> dict[str, dict[str, float]]:
    parameters: dict[str, dict[str, float]] = {}

    for idx, variable in enumerate(FUZZY_VARIABLES):
        spec: VariableSpec = VARIABLE_SPECS[variable]

        lo: float = spec["min"]
        hi: float = spec["max"]

        raw: FloatArray = np.sort(np.clip(genes[idx * 3 : idx * 3 + 3], 0.0, 1.0))
        min_sep: float = (hi - lo) * MIN_SEPARATION_FRACTION

        points: list[float] = [lo + float(raw[0]) * (hi - lo)]

        for value in raw[1:]:
            next_point: float = lo + float(value) * (hi - lo)
            points.append(max(next_point, points[-1] + min_sep))

        points[2] = min(points[2], hi)
        points[1] = min(points[1], points[2] - min_sep)
        points[0] = min(points[0], points[1] - min_sep)

        parameters[variable] = {
            "p1": round(points[0], 2),
            "p2": round(points[1], 2),
            "p3": round(points[2], 2),
        }

    return parameters


def genes_from_parameters(parameters: dict[str, dict[str, float]]) -> FloatArray:
    genes: list[float] = []

    for variable in FUZZY_VARIABLES:
        spec: VariableSpec = VARIABLE_SPECS[variable]

        lo: float = spec["min"]
        hi: float = spec["max"]

        for key in ("p1", "p2", "p3"):
            genes.append((parameters[variable][key] - lo) / (hi - lo))

    return np.array(genes, dtype=float)


class OptimizationResult:
    def __init__(
        self,
        best_genes: FloatArray,
        best_parameters: dict[str, dict[str, float]],
        best_fitness: float,
        history: list[dict[str, float]],
        validation_metrics: ClassificationMetrics | None,
    ) -> None:
        self.best_genes = best_genes
        self.best_parameters = best_parameters
        self.best_fitness = best_fitness
        self.history = history
        self.validation_metrics = validation_metrics


def mean_fitness(population: list[Any]) -> float:
    if not population:
        return 0.0

    total: float = 0.0

    for individual in population:
        total += float(individual.fitness.values[0])

    return total / len(population)


class GeneticOptimizer:
    def __init__(
        self,
        population_size: int = POPULATION_SIZE,
        generations: int = GENERATIONS,
        elite_count: int = ELITE_COUNT,
        crossover_rate: float = CROSSOVER_RATE,
        mutation_rate: float = MUTATION_RATE,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.elite_count = elite_count
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed

    def fitness(
        self, genes: FloatArray, values: dict[str, FloatArray], y_true: NDArray[np.int64]
    ) -> float:
        system: FuzzySystem = FuzzySystem(decode_genes(genes))

        risk, _, _ = system.evaluate_batch(values)

        y_pred: NDArray[np.int64] = (risk >= RISK_THRESHOLD).astype(np.int64)

        return evaluate_predictions(y_true, y_pred).balanced_accuracy

    def _fitness_values(
        self, individual: list[float], values: dict[str, FloatArray], y_true: NDArray[np.int64]
    ) -> tuple[float, ...]:
        return (self.fitness(np.asarray(individual, dtype=float), values, y_true),)

    def _sample(
        self, values: dict[str, FloatArray], y_true: NDArray[np.int64]
    ) -> tuple[dict[str, FloatArray], NDArray[np.int64]]:
        n: int = len(y_true)

        if FITNESS_SAMPLE_SIZE <= 0 or n <= FITNESS_SAMPLE_SIZE:
            return values, y_true

        rng: np.random.Generator = np.random.default_rng(self.seed)

        positives: NDArray[np.int64] = np.flatnonzero(y_true == 1)
        negatives: NDArray[np.int64] = np.flatnonzero(y_true == 0)

        sample: NDArray[np.int64] = positives.copy()
        remaining: int = FITNESS_SAMPLE_SIZE - len(positives)

        if remaining > 0:
            chosen: NDArray[np.int64] = rng.choice(
                negatives, size=min(remaining, len(negatives)), replace=False
            )
            sample = np.concatenate([sample, chosen])

        sample = np.sort(sample)

        sampled_values: dict[str, FloatArray] = {}

        for variable in FUZZY_VARIABLES:
            sampled_values[variable] = values[variable][sample]

        return sampled_values, y_true[sample]

    def run(
        self,
        values_train: dict[str, FloatArray],
        y_train: NDArray[np.int64],
        values_val: dict[str, FloatArray] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> OptimizationResult:
        random.seed(self.seed)

        values_fit, y_fit = self._sample(values_train, y_train)

        gene_random = partial(random.uniform, *GENE_RANGE)

        population: list[Any] = []

        for _ in range(self.population_size):
            population.append(tools.initRepeat(Individual, gene_random, N_GENES))

        population[0][:] = genes_from_parameters(DEFAULT_PARAMETERS)

        for individual in population:
            individual.fitness.values = self._fitness_values(individual, values_fit, y_fit)

        hall_of_fame = tools.HallOfFame(self.elite_count)
        hall_of_fame.update(population)

        best_genes: FloatArray = np.asarray(hall_of_fame[0], dtype=float)
        best_fitness: float = float(hall_of_fame[0].fitness.values[0])

        history: list[dict[str, float]] = [
            {
                "generation": 0,
                "best": best_fitness,
                "mean": mean_fitness(population),
            }
        ]

        for generation in range(1, self.generations + 1):
            elites = tools.selBest(population, self.elite_count)
            offspring = tools.selTournament(
                population,
                k=self.population_size - self.elite_count,
                tournsize=TOURNAMENT_SIZE,
            )

            cloned_offspring: list[Any] = []

            for individual in offspring:
                cloned_offspring.append(deepcopy(individual))

            offspring = cloned_offspring

            for child_a, child_b in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    tools.cxOnePoint(child_a, child_b)
                tools.mutGaussian(child_a, 0.0, 0.1, self.mutation_rate)
                tools.mutGaussian(child_b, 0.0, 0.1, self.mutation_rate)
                del child_a.fitness.values
                del child_b.fitness.values

            for individual in offspring:
                if not individual.fitness.valid:
                    individual.fitness.values = self._fitness_values(individual, values_fit, y_fit)

            population = elites + offspring
            hall_of_fame.update(population)

            generation_best: float = float(hall_of_fame[0].fitness.values[0])

            if generation_best > best_fitness:
                best_fitness = generation_best
                best_genes = np.asarray(hall_of_fame[0], dtype=float)

            history.append(
                {
                    "generation": generation,
                    "best": best_fitness,
                    "mean": mean_fitness(population),
                }
            )

        best_parameters: dict[str, dict[str, float]] = decode_genes(best_genes)

        result: OptimizationResult = OptimizationResult(
            best_genes=best_genes,
            best_parameters=best_parameters,
            best_fitness=best_fitness,
            history=history,
            validation_metrics=None,
        )

        if values_val is not None and y_val is not None:
            system: FuzzySystem = FuzzySystem(best_parameters)

            risk_val, _, _ = system.evaluate_batch(values_val)

            y_pred_val: NDArray[np.int64] = (risk_val >= RISK_THRESHOLD).astype(np.int64)

            result.validation_metrics = evaluate_predictions(y_val, y_pred_val)

        return result