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

    def set_initial_population(self):
        for i in range(POPULATION_SIZE):
            self.population[i] = self.rng.integers(0, 2, size=6, dtype=numpy.byte)

    def evaluate_fitness(self):
        for i in range(POPULATION_SIZE):
            self.objetives[i] = self.evaluate_individual(self.population[i])

        for i in range(POPULATION_SIZE):
            print(f"Objetives {i}: {self.objetives[i]}")

    def evaluate_individual(self, individual: NDArray) -> tuple[float, float]:
        return (self.room.lighting(individual), self.room.cost(individual))
