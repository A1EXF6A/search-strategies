import numpy
from algorithms.genetic import GeneticAlgorithm
from algorithms.local_search import LocalSearch
from utils.room import Room

RUNS = 5

OPTIMAL_GLOBAL = (1, 0, 1, 1, 0, 1)


def _run_local_search(runs: int) -> int:
    hits: int = 0

    for _ in range(runs):
        result: dict = LocalSearch().solve()

        if tuple(result["focos"]) == OPTIMAL_GLOBAL:
            hits += 1

    return hits


def _run_genetic(runs: int, elitism: bool) -> tuple[int, int]:
    recommended_hits: int = 0
    front_hits: int = 0

    for _ in range(runs):
        algorithm: GeneticAlgorithm = GeneticAlgorithm(elitism=elitism)

        result: dict = algorithm.solve()

        if tuple(result["mejor_escalarizacion"]["focos"]) == OPTIMAL_GLOBAL:
            recommended_hits += 1

        if any(
            tuple(solution["focos"]) == OPTIMAL_GLOBAL
            for solution in result["frente"]
        ):
            front_hits += 1

    return recommended_hits, front_hits


def run_benchmark(runs: int = RUNS) -> dict:
    room: Room = Room(0.5)

    optimal_scalar: float = (
        room.lighting(numpy.array(OPTIMAL_GLOBAL, dtype=numpy.byte))
        - 10 * room.cost(numpy.array(OPTIMAL_GLOBAL, dtype=numpy.byte))
    )

    local_search_hits: int = _run_local_search(runs)

    ga_recommended, ga_front = _run_genetic(runs, elitism=False)

    elite_recommended, elite_front = _run_genetic(runs, elitism=True)

    return {
        "corridas": runs,
        "optimo_global": list(OPTIMAL_GLOBAL),
        "escalar_optimo": float(optimal_scalar),
        "algoritmos": {
            "busqueda_local": {
                "nombre": "Búsqueda local",
                "encontro_optimo": local_search_hits,
                "porcentaje": round(local_search_hits / runs * 100, 1),
            },
            "ga_puro": {
                "nombre": "GA puro (estocástico)",
                "encontro_optimo_recomendada": ga_recommended,
                "recomendada_porcentaje": round(ga_recommended / runs * 100, 1),
                "encontro_optimo_frente": ga_front,
                "frente_porcentaje": round(ga_front / runs * 100, 1),
            },
            "ga_elitista": {
                "nombre": "GA con elitismo",
                "encontro_optimo_recomendada": elite_recommended,
                "recomendada_porcentaje": round(elite_recommended / runs * 100, 1),
                "encontro_optimo_frente": elite_front,
                "frente_porcentaje": round(elite_front / runs * 100, 1),
            },
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_benchmark(), indent=2))
