from __future__ import annotations

import os
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

    print(f"\n{YELLOW}--- INFORMACION ---{RESET}\n")

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

    os.system("clear" if os.name == "posix" else "cls")

    optimal_path: list[Node]
    all_solutions: list[list[Node]]
    tree: Tree
    metrics: Metrics

    optimal_path, all_solutions, tree, metrics = strategy.run(
        agent_position, GOAL_POSITION, MAX_X, MAX_Y
    )

    # print(f"{GREEN}--- ARBOL DE BUSQUEDA ---{RESET}")

    # tree.print(optimal_path, goal_nodes)

    print(f"\n{GREEN}--- SOLUCIONES ENCONTRADAS ---{RESET}")

    goal_nodes: list[Node] = []
    for sol in all_solutions:
        if sol:
            goal_nodes.append(sol[-1])

    print(f"\nTotal de soluciones encontradas: {len(all_solutions)}")

    if len(all_solutions) > 1:
        print()
        print(f"{YELLOW}--- OTRAS SOLUCIONES ---{RESET}")

        counter: int = 1
        for i, sol in enumerate(all_solutions):
            if i == 0:
                continue

            print(f"\n--- Solucion alternativa {counter} ---")
            print_path(sol)
            counter += 1

    print()
    print(f"{GREEN}--- SOLUCION OPTIMA ---{RESET}")
    print()

    print_path(optimal_path)

    if not optimal_path:
        print("\nNo se encontro ninguna solucion.")
        return

    print()
    print(f"{GREEN}--- MAPA ---{RESET}")
    print()

    print(f"Posicion del agente: {agent_position}")
    print(f"Posicion del objetivo: {GOAL_POSITION}")

    print_map(agent_position, GOAL_POSITION)

    print_metrics(strategy, metrics, optimal_path)


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
    print()
    print(f"Algoritmo: {strategy_name}")
    print(f"Estructura utilizada: {strategy.data_structure_used()}")
    print(f"Profundidad: {path[-1].depth}")
    print(f"Movimientos realizados: {len(path) - 1}")
    print(f"Nodos expandidos: {metrics.expanded_nodes}")
    print()
    print(f"Complejidad en tiempo: {metrics.execution_time:.6f} segundos")
    print()
    print(f"Tamano estimado por nodo: {Metrics.bytes_per_node} bytes")

    print(f"Total de nodos creados: {metrics.total_nodes}")
    print(
        f"Formula: {metrics.total_nodes} nodos x {Metrics.bytes_per_node} bytes = {metrics.memory_bytes()} bytes = {metrics.memory_mib():.6f} MiB"
    )


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
