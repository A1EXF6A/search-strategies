from __future__ import annotations

import random

from estrategias.base import Strategy
from estrategias.bfs_strategy import BFSStrategy
from estrategias.dfs_strategy import DFSStrategy
from estrategias.metrics import Metrics
from estrategias.node import Node
from estrategias.tree import Tree

MAX_X: int = 10
MAX_Y: int = 20

GOAL_POSITION: tuple[int, int] = (1, 10)

RESET: str = "\033[0m"
GREEN: str = "\033[92m"
YELLOW: str = "\033[93m"


def main() -> None:
    print("--- Agente de Búsqueda ---")

    print("\n--- ¿Dónde empezará la búsqueda? ---")
    print("1. Ubicacion aleatoria")
    print("2. Ubicacion especifica")

    option: str = input("Elige una opcion (1-2): ")

    agent_position: tuple[int, int] = (0, 0)

    if option == "1":
        agent_position = (random.randint(1, MAX_X), random.randint(1, MAX_Y))
    elif option == "2":
        x: int = int(input(f"Ingrese la coordenada X (0-{MAX_X}): "))
        y: int = int(input(f"Ingrese la coordenada Y (0-{MAX_Y}): "))
        agent_position = (x, y)
    else:
        agent_position = (random.randint(0, MAX_X), random.randint(0, MAX_Y))

    print("\n--- Resumen de la búsqueda ---")

    print(f"Posicion del agente: {agent_position}")
    print(f"Posicion del objetivo: {GOAL_POSITION}")

    print("\nMapa:")

    print_map(agent_position, GOAL_POSITION)

    print("\n--- Que estrategia se utilizara? ---")
    print("1. Búsqueda por anchura")
    print("2. Búsqueda por profundidad")
    print("3. Búsqueda iterativa")
    print("4. Búsqueda bidireccional")

    strategy_option: str = input("Elige una opcion (1-4): ")

    strategy: Strategy | None = None

    if strategy_option == "1":
        strategy = BFSStrategy()
    elif strategy_option == "2":
        strategy = DFSStrategy()
    elif strategy_option == "3":
        print("Búsqueda iterativa no implementada.")
        return
    elif strategy_option == "4":
        print("Búsqueda bidireccional no implementada.")
        return
    else:
        print("Opcion no valida. Intenta de nuevo.")
        return

    optimal_path: list[Node]
    all_solutions: list[list[Node]]
    tree: Tree
    metrics: Metrics

    optimal_path, all_solutions, tree, metrics = strategy.run(
        agent_position, GOAL_POSITION, MAX_X, MAX_Y
    )

    print(f"{GREEN}--- ARBOL DE BUSQUEDA ---{RESET}")

    goal_nodes: list[Node] = []
    for sol in all_solutions:
        if sol:
            goal_nodes.append(sol[-1])

    tree.print(optimal_path, goal_nodes)

    print(f"\nTotal de soluciones encontradas: {len(all_solutions)}")

    if not optimal_path:
        print("\nNo se encontro ninguna solucion.")
        return

    print()
    print(f"{GREEN}--- MAPA ---{RESET}")
    print_map(agent_position, GOAL_POSITION)

    print()
    print(f"{GREEN}--- SOLUCION OPTIMA ---{RESET}")

    print_path(optimal_path)
    print_metrics(strategy, metrics, optimal_path)

    if len(all_solutions) > 1:
        print("\n" + "=" * 50)
        print("OTRAS SOLUCIONES")
        print("=" * 50)

        counter: int = 1
        for i, sol in enumerate(all_solutions):
            if i == 0:
                continue

            print(f"\n--- Solucion alternativa {counter} ---")
            print_path(sol)
            counter += 1


def print_path(path: list[Node]) -> None:
    for i, node in enumerate(path):
        print(f"Paso {i}: Posicion ({node.position[0]}, {node.position[1]})")
        if node.action is not None:
            print(f"  Accion: {node.action}")


def print_metrics(strategy: Strategy, metrics: Metrics, path: list[Node]) -> None:
    strategy_name: str = ""

    if isinstance(strategy, BFSStrategy):
        strategy_name = "Búsqueda por anchura"
    elif isinstance(strategy, DFSStrategy):
        strategy_name = "Búsqueda por profundidad"

    print()
    print(f"{GREEN}--- METRICAS ---{RESET}")
    print(f"Algoritmo: {strategy_name}")
    print(f"Estructura utilizada: {strategy.data_structure_used()}")
    print(f"Profundidad: {path[-1].depth}")
    print(f"Cantidad de pasos: {len(path) - 1}")
    print(f"Estados en el camino: {len(path)}")
    print(f"Complejidad en tiempo: {metrics.expanded_nodes}")
    print(f"Complejidad en espacio: {metrics.space()}")
    print(f"Es optima: {'Si' if strategy.is_optimal() else 'No'}")


def print_map(agent_position: tuple[int, int], goal_position: tuple[int, int]) -> None:
    print(f"   +{'-' * 39}+")

    agent_in_y: bool = False
    goal_in_y: bool = False

    for i in range(MAX_Y, 0, -1):
        print(f"{str(i).ljust(3)}|", end="")

        agent_in_y = agent_position[1] == i
        goal_in_y = goal_position[1] == i

        for j in range(MAX_X):
            if agent_in_y and (agent_position[0] - 1) == j:
                print(" A |", end="")
                continue

            if goal_in_y and (goal_position[0] - 1) == j:
                print(" G |", end="")
                continue

            print("   |", end="")

            if j == MAX_X - 1:
                print()

    print(f"   +{'-' * 39}+")

    print("     ", end="")

    for i in range(1, MAX_X + 1):
        print(f"{i}   ", end="")

    print()
    print("A = Agente")
    print("G = Objetivo")


if __name__ == "__main__":
    main()
