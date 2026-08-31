import numpy
from numpy.typing import NDArray
from utils.room import Room

POPULATION_SIZE = 20

UNIFORM_WEIGHT = 0.5


class GeneticAlgorithm:
    population: NDArray
    objetives: NDArray

    children: NDArray

    rng: numpy.random.Generator

    room: Room

    def run(self):
        self.rng = numpy.random.default_rng()

        self.population = numpy.zeros((POPULATION_SIZE, 6), dtype=numpy.byte)
        self.objetives = numpy.zeros((POPULATION_SIZE, 2), dtype=numpy.float32)

        self.children = numpy.zeros((2, 6), dtype=numpy.byte)

        self.room = Room(UNIFORM_WEIGHT)

        # 1. Define initial population

        self.set_initial_population()

        # 1.1 Evaluate fitness of each individual in the population

        self.evaluate_fitness()

        # 1.2 Apply Pareto ranking to the population

        ranking: NDArray = self.pareto_ranking(self.objetives)

        total_ranking: float = 0.0

        for i in range(POPULATION_SIZE):
            total_ranking += POPULATION_SIZE - (ranking[i] - 1)

        # 2. Select parents for crossover

        # 2.1 Define probability distribution based on Pareto ranking

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
