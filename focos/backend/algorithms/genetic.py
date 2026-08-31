import numpy
from numpy.typing import NDArray
from utils.room import Room

POPULATION_SIZE = 20

UNIFORM_WEIGHT = 0.5

MUTATION_PROBABILITY = 0.3


class GeneticAlgorithm:
    population: NDArray
    objetives: NDArray

    children: NDArray

    split: int

    roulette: NDArray

    last_mutation_individual: int
    last_mutation_foco: int

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

        # 2.2 Build the roulette wheel: an array of 100 elements where each
        # individual appears as many times as its probability dictates,
        # distributed randomly across the array.

        self.build_roulette(probability)

        # 2.3 Select parents: pick two random indices into the roulette.

        parent_1: int
        parent_2: int

        parent_1, parent_2 = self.select_parents()

        # 3. Crossover: apply single-point crossover to produce two children

        children: NDArray = self.crossover(parent_1, parent_2)

        print(f"  Hijo 1: {children[0]}")
        print(f"  Hijo 2: {children[1]}")

        # 4. Mutation: with probability MUTATION_PROBABILITY, flip one foco

        self.mutate()

        if self.last_mutation_individual == -1:
            print(f"Mutación (probabilidad {MUTATION_PROBABILITY}): no hubo mutación")
        else:
            print(
                f"Mutación (probabilidad {MUTATION_PROBABILITY}): mutó el individuo "
                f"{self.last_mutation_individual} (foco índice {self.last_mutation_foco})"
            )

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
