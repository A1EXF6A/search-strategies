import numpy
from numpy.typing import NDArray
from utils.room import Room

UNIFORM_WEIGHT = 0.5

WEIGHT_COST = 10


class LocalSearch:
    current_state: NDArray
    current_fitness: float

    best_state: NDArray
    best_fitness: float

    steps: int

    rng: numpy.random.Generator

    room: Room

    def run(self):
        self.rng = numpy.random.default_rng()

        self.room = Room(UNIFORM_WEIGHT)

        self.steps = 0

        self.initialize()

        print(f"Estado inicial (aleatorio): {self.current_state} "
              f"-> fitness {self.current_fitness:.4f}")

        # Hill climbing: keep moving to the best neighbor until no neighbour
        # yields a better fitness value (local optimum reached).

        self.hill_climb()

        print(f"Búsqueda local terminada en {self.steps} pasos")
        print(f"Mejor estado encontrado: {self.best_state} "
              f"-> fitness {self.best_fitness:.4f}")

    def solve(self) -> dict:
        self.rng = numpy.random.default_rng()

        self.room = Room(UNIFORM_WEIGHT)

        self.steps = 0

        self.initialize()

        self.hill_climb(verbose=False)

        return self.result_dict()

    def result_dict(self) -> dict:
        individual: NDArray = self.best_state

        metrics: dict[str, float] = self.room.metrics(individual)

        focos: list[int] = [int(value) for value in individual]

        return {
            "focos": focos,
            "focos_encendidos": int(numpy.sum(individual)),
            "iluminacion": float(metrics["lighting"]),
            "iluminacion_promedio": float(metrics["avg_lighting"]),
            "iluminacion_minima": float(metrics["min_lighting"]),
            "uniformidad": float(metrics["uniformity"]),
            "costo": float(metrics["cost"]),
            "potencia_w": 15.0,
            "tiempo_h": 5.0,
            "energia_kwh": float(metrics["consumed_energy"]),
            "costo_por_foco": float(metrics["cost_per_lightbulb"]),
        }

    def initialize(self) -> None:
        self.current_state = numpy.zeros(6, dtype=numpy.byte)

        for i in range(self.current_state.shape[0]):
            self.current_state[i] = self.rng.integers(0, 2)

        self.current_fitness = self.evaluate(self.current_state)

        self.best_state = self.current_state.copy()
        self.best_fitness = self.current_fitness

    def evaluate(self, individual: NDArray) -> float:
        lighting: float = self.room.lighting(individual)
        cost: float = self.room.cost(individual)

        return lighting - WEIGHT_COST * cost

    def get_neighbors(self, individual: NDArray) -> NDArray:
        neighbors: NDArray = numpy.zeros(
            (individual.shape[0], individual.shape[0]), dtype=numpy.byte
        )

        for i in range(individual.shape[0]):
            neighbors[i] = individual.copy()
            neighbors[i, i] = 1 - neighbors[i, i]

        return neighbors

    def hill_climb(self, verbose: bool = True) -> None:
        improved: bool = True

        while improved:
            improved = False

            best_neighbor_index: int = -1
            best_neighbor_fitness: float = self.current_fitness

            neighbors: NDArray = self.get_neighbors(self.current_state)

            for i in range(neighbors.shape[0]):
                neighbor_fitness: float = self.evaluate(neighbors[i])

                if neighbor_fitness > self.current_fitness and neighbor_fitness > best_neighbor_fitness:
                    best_neighbor_fitness = neighbor_fitness
                    best_neighbor_index = i

            if best_neighbor_index != -1:
                best_neighbor: NDArray = neighbors[best_neighbor_index]

                if verbose:
                    print(f"  Paso {self.steps + 1}: estado {self.current_state} "
                          f"(fitness {self.current_fitness:.4f}) -> "
                          f"mejor vecino {best_neighbor} (fitness {best_neighbor_fitness:.4f})")

                self.current_state = best_neighbor
                self.current_fitness = best_neighbor_fitness
                self.steps += 1

                if self.current_fitness > self.best_fitness:
                    self.best_state = self.current_state.copy()
                    self.best_fitness = self.current_fitness

                improved = True
