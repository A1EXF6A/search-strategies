import numpy
from numpy.typing import NDArray
from utils.room import Room

POPULATION_SIZE = 20

UNIFORM_WEIGHT = 0.5

MUTATION_PROBABILITY = 0.3

MAX_EPOCHS = 100

PATIENCE = 32

MAX_FRONT_SOLUTIONS = 6

SURVIVAL_FITNESS_WEIGHT_COST = 10


class GeneticAlgorithm:
    population: NDArray
    objetives: NDArray

    children: NDArray

    split: int

    roulette: NDArray

    last_mutation_individual: int
    last_mutation_foco: int

    epoch: int
    patience_without_improvement: int
    previous_front: NDArray
    best_front: NDArray
    stop_reason: str

    rng: numpy.random.Generator

    room: Room

    def __init__(self, elitism: bool = False) -> None:
        self.elitism = elitism

    def run(self):
        self.evolve(verbose=True)

    def solve(self) -> dict:
        self.evolve(verbose=False)

        return self.result_dict()

    def evolve(self, verbose: bool = True) -> None:
        self.rng = numpy.random.default_rng()

        self.population = numpy.zeros((POPULATION_SIZE, 6), dtype=numpy.byte)
        self.objetives = numpy.zeros((POPULATION_SIZE, 2), dtype=numpy.float32)

        self.children = numpy.zeros((2, 6), dtype=numpy.byte)

        self.room = Room(UNIFORM_WEIGHT)

        self.epoch = 0
        self.patience_without_improvement = 0

        # 1. Define initial population

        self.set_initial_population()

        # 1.1 Evaluate fitness of each individual in the population

        self.evaluate_fitness()

        # 1.2 Apply Pareto ranking to the population

        ranking: NDArray = self.pareto_ranking(self.objetives)

        # Save the initial Pareto front as a reference

        self.previous_front = self.pareto_front(ranking)
        self.best_front = self.previous_front.copy()

        if verbose:
            print(f"Frente Pareto inicial (época {self.epoch}): {self.previous_front}")

        # 2-6. Evolution loop

        while not self.stopping_condition():
            self.epoch += 1

            # Elitism: keep a copy of the most fit individual of the current
            # generation so it cannot be lost by crossover/mutation/replacement.

            elite: NDArray | None = None

            if self.elitism:
                elite = self.population[self.best_scalarized_index()].copy()

            # 2. Select parents based on Pareto ranking

            ranking = self.pareto_ranking(self.objetives)

            probability: NDArray = self.compute_probabilities(ranking)

            self.build_roulette(probability)

            parent_1: int
            parent_2: int

            parent_1, parent_2 = self.select_parents()

            # 3. Crossover: apply single-point crossover to produce two children

            self.crossover(parent_1, parent_2)

            # 4. Mutation: with probability MUTATION_PROBABILITY, flip one foco

            self.mutate()

            # 5. Replacement: remove two individuals at random from the 22
            #    candidates (20 population + 2 children) to go back to 20

            self.replacement()

            # 5.1 Elitism: reinsert the preserved elite, replacing a random
            #     individual, so the best solution always survives.

            if self.elitism and elite is not None:
                victim: int = int(self.rng.integers(0, self.population.shape[0]))
                self.population[victim] = elite

            # 5.2 Re-evaluate fitness of the new population

            self.evaluate_fitness()

            # 6. Detention: recompute the Pareto front and compare with the
            #    previous generation; update patience.

            self.check_improvement()

            if verbose:
                print(
                    f"Época {self.epoch} finalizada - paciencia: "
                    f"{self.patience_without_improvement}"
                )

        print(f"Algoritmo detenido en época {self.epoch}: {self.stop_reason}")
        print(f"Mejor frente Pareto encontrado: {self.best_front}")

    def compute_probabilities(self, ranking: NDArray) -> NDArray:
        total_ranking: float = 0.0

        for i in range(POPULATION_SIZE):
            total_ranking += POPULATION_SIZE - (ranking[i] - 1)

        probability: NDArray = numpy.zeros(POPULATION_SIZE, dtype=numpy.int64)

        for i in range(POPULATION_SIZE):
            probability[i] = round(
                ((POPULATION_SIZE - (ranking[i] - 1)) / total_ranking) * 100
            )

        if probability.sum() > 100:
            rnd_index = self.rng.integers(0, POPULATION_SIZE)
            probability[rnd_index] -= probability.sum() - 100

        if probability.sum() < 100:
            rnd_index = self.rng.integers(0, POPULATION_SIZE)
            probability[rnd_index] += 100 - probability.sum()

        return probability

    def build_roulette(self, probability: NDArray) -> None:
        self.roulette = numpy.zeros(100, dtype=numpy.int64)

        offset: int = 0

        for i in range(POPULATION_SIZE):
            count: int = int(probability[i])

            self.roulette[offset : offset + count] = i

            offset += count

        self.rng.shuffle(self.roulette)

    def select_parents(self) -> tuple[int, int]:
        index_1: int = int(self.rng.integers(0, self.roulette.size))
        parent_1: int = int(self.roulette[index_1])

        index_2: int = int(self.rng.integers(0, self.roulette.size))
        parent_2: int = int(self.roulette[index_2])

        while parent_2 == parent_1:
            index_2 = int(self.rng.integers(0, self.roulette.size))
            parent_2 = int(self.roulette[index_2])

        return parent_1, parent_2

    def crossover(self, parent_1: int, parent_2: int) -> NDArray:
        self.split = int(self.rng.integers(1, self.population.shape[1]))

        first: NDArray = self.population[parent_1]
        second: NDArray = self.population[parent_2]

        self.children[0] = numpy.concatenate(
            (first[: self.split], second[self.split :])
        )
        self.children[1] = numpy.concatenate(
            (second[: self.split], first[self.split :])
        )

        return self.children

    def mutate(self) -> None:
        draw: float = self.rng.random()

        if draw < MUTATION_PROBABILITY:
            individual_index: int = int(
                self.rng.integers(0, self.population.shape[0] + self.children.shape[0])
            )

            if individual_index < self.population.shape[0]:
                target: NDArray = self.population[individual_index]
            else:
                target = self.children[individual_index - self.population.shape[0]]

            foco_index: int = int(self.rng.integers(0, self.population.shape[1]))

            target[foco_index] = 1 - target[foco_index]

            self.last_mutation_individual = individual_index
            self.last_mutation_foco = foco_index
        else:
            self.last_mutation_individual = -1
            self.last_mutation_foco = -1

    def replacement(self) -> None:
        candidates: NDArray = numpy.concatenate((self.population, self.children))

        first_removed: int = int(self.rng.integers(0, candidates.shape[0]))

        second_removed: int = int(self.rng.integers(0, candidates.shape[0]))
        while second_removed == first_removed:
            second_removed = int(self.rng.integers(0, candidates.shape[0]))

        survivors: NDArray = numpy.delete(
            candidates, [first_removed, second_removed], axis=0
        )

        self.population = survivors.copy()

    def pareto_front(self, ranking: NDArray) -> NDArray:
        indices: NDArray = numpy.where(ranking == 1)[0]
        return self.objetives[indices].copy()

    def check_improvement(self) -> None:
        ranking: NDArray = self.pareto_ranking(self.objetives)
        current_front: NDArray = self.pareto_front(ranking)

        improved: bool = self.front_improved(self.previous_front, current_front)

        if improved:
            self.patience_without_improvement = 0
            self.best_front = current_front.copy()
        else:
            self.patience_without_improvement += 1

        self.previous_front = current_front.copy()

    def front_improved(self, previous_front: NDArray, current_front: NDArray) -> bool:
        for current_solution in current_front:
            for previous_solution in previous_front:
                if self.dominates(current_solution, previous_solution):
                    return True

        return False

    def dominates(self, a: NDArray, b: NDArray) -> bool:
        lighting_a: float = float(a[0])
        cost_a: float = float(a[1])
        lighting_b: float = float(b[0])
        cost_b: float = float(b[1])

        return (lighting_a >= lighting_b and cost_a <= cost_b) and (
            lighting_a > lighting_b or cost_a < cost_b
        )

    def stopping_condition(self) -> bool:
        if self.patience_without_improvement >= PATIENCE:
            self.stop_reason = "Paciencia agotada"
            return True

        if self.epoch >= MAX_EPOCHS:
            self.stop_reason = "Épocas agotadas"
            return True

        return False

    def set_initial_population(self):
        for i in range(POPULATION_SIZE):
            self.population[i] = self.rng.integers(0, 2, size=6, dtype=numpy.byte)

    def evaluate_fitness(self):
        for i in range(POPULATION_SIZE):
            self.objetives[i] = self.evaluate_individual(self.population[i])

    def evaluate_individual(self, individual: NDArray) -> tuple[float, float]:
        return (self.room.lighting(individual), self.room.cost(individual))

    def pareto_ranking(self, pop: NDArray) -> NDArray:
        n: int = pop.shape[0]
        rank: NDArray = numpy.zeros(n, dtype=int)
        remaining: NDArray = numpy.ones(n, dtype=bool)
        current_rank: int = 1

        while numpy.any(remaining):
            index: NDArray = numpy.where(remaining)[0]
            sub_population: NDArray = pop[index]
            k: int = len(index)

            if k == 1:
                rank[index[0]] = current_rank
                remaining[index[0]] = False
                current_rank += 1
                continue

            f1: NDArray = sub_population[:, 0][:, None]
            f2: NDArray = sub_population[:, 1][:, None]

            cond1: NDArray = f1 >= f1.T
            cond2: NDArray = f2 <= f2.T
            strict: NDArray = (f1 > f1.T) | (f2 < f2.T)
            dominates: NDArray = cond1 & cond2 & strict

            dominated: NDArray = numpy.any(dominates, axis=0)
            non_dominated: NDArray = ~dominated

            rank[index[non_dominated]] = current_rank
            remaining[index[non_dominated]] = False
            current_rank += 1

        return rank

    def result_dict(self) -> dict:
        ranking: NDArray = self.pareto_ranking(self.objetives)
        front_indices: NDArray = numpy.where(ranking == 1)[0]

        solutions: list[dict] = []

        seen: set[str] = set()

        for index in front_indices:
            individual: NDArray = self.population[index]

            key: str = "".join(str(int(v)) for v in individual)

            if key in seen:
                continue

            seen.add(key)

            metrics: dict[str, float] = self.room.metrics(individual)

            solutions.append(
                {
                    "focos": [int(value) for value in individual],
                    "iluminacion": float(metrics["lighting"]),
                    "iluminacion_promedio": float(metrics["avg_lighting"]),
                    "iluminacion_minima": float(metrics["min_lighting"]),
                    "uniformidad": float(metrics["uniformity"]),
                    "costo": float(metrics["cost"]),
                    "focos_encendidos": int(numpy.sum(individual)),
                }
            )

        if len(solutions) > MAX_FRONT_SOLUTIONS:
            indices: NDArray = self.rng.choice(
                numpy.arange(len(solutions)),
                size=MAX_FRONT_SOLUTIONS,
                replace=False,
            )

            solutions = [solutions[i] for i in indices]

        best_scalarized_index: int = self.best_scalarized_index()

        best_scalarized_metrics: dict[str, float] = self.room.metrics(
            self.population[best_scalarized_index]
        )

        return {
            "frente": solutions,
            "total_soluciones": len(solutions),
            "mejor_escalarizacion": {
                "focos": [int(value) for value in self.population[best_scalarized_index]],
                "focos_encendidos": int(
                    numpy.sum(self.population[best_scalarized_index])
                ),
                "iluminacion": float(best_scalarized_metrics["lighting"]),
                "iluminacion_promedio": float(
                    best_scalarized_metrics["avg_lighting"]
                ),
                "iluminacion_minima": float(
                    best_scalarized_metrics["min_lighting"]
                ),
                "uniformidad": float(best_scalarized_metrics["uniformity"]),
                "costo": float(best_scalarized_metrics["cost"]),
            },
            "potencia_w": 15.0,
            "tiempo_h": 5.0,
        }

    def best_scalarized_index(self) -> int:
        best_index: int = 0
        best_value: float = -numpy.inf

        for i in range(self.population.shape[0]):
            lighting: float = self.room.lighting(self.population[i])
            cost: float = self.room.cost(self.population[i])

            value: float = lighting - SURVIVAL_FITNESS_WEIGHT_COST * cost

            if value > best_value:
                best_value = value
                best_index = i

        return best_index
