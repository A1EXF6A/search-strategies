from __future__ import annotations

import os

from algorithms.astar import solve_a_star
from algorithms.greedy import solve_greedy
from structures.metrics import Metrics
from structures.node import Node
from structures.nodea import NodeA
from structures.tree import Tree

RED: str = "\033[0m"
BLUE: str = "\033[1m"
GREEN: str = "\033[92m"
YELLOW: str = "\033[93m"
CYAN: str = "\033[96m"
RESET: str = "\033[0m"

SEP_DOUBLE: str = "═" * 68
SEP_SIMPLE: str = "─" * 68


def print_board(board: tuple[tuple[int, ...], ...], prefix: str = "  ") -> None:
    print(prefix + "┌───────────┐")
    for row in board:
        cells: str = "│".join(
            f" {YELLOW}□{RED} " if x == 0 else f" {BLUE}{x}{RED} " for x in row
        )
        print(prefix + "│" + cells + "│")
    print(prefix + "└───────────┘")


def main() -> None:
    print(f"{YELLOW}--- INFORMACION ---{RESET}\n")

    print(f"{BLUE}Estado Inicial: A{RED}")
    print(f"{BLUE}Estado Objetivo: K{RED}")

    print(f"\n{BLUE}--- ¿Qué algoritmo desea utilizar? ---{RED}")
    print("1. A*")
    print("2. Voraz")

    option: str = input("Elige una opcion (1-2): ")

    os.system("clear" if os.name == "posix" else "cls")

    solve_fn = solve_a_star

    if option == "2":
        solve_fn = solve_greedy

    initial_state: str = "A"
    goal_state: str = "K"
    goal_node, root, metrics = solve_fn(initial_state, goal_state)
    path: list[Node | NodeA] = goal_node.path() if goal_node else []

    print()
    print(f"{GREEN}--- ARBOL DE BUSQUEDA ---{RESET}")

    optimal_set: set[Node | NodeA] = set(path)
    goals: set[Node | NodeA] = {path[-1]} if path else set()
    Tree(root).print_limited(
        optimal_path=list(optimal_set),
        goals=list(goals),
        max_depth=3,
    )

    print()
    print(f"{GREEN}--- SOLUCION ENCONTRADA ---{RESET}")
    print()

    steps: int = len(path) - 1 if path else 0

    if not path:
        print(f"  {CYAN}No hay solucion.{RED}")
        return

    for i, node in enumerate(path):
        print(f"{SEP_SIMPLE}")
        print(f"ID nodo: {node.id}  |  h={node.h}  f={node.f}")

    print()
    print(f"{GREEN}--- METRICAS ---{RESET}")
    print()
    print(f"Algoritmo: {metrics.algorithm}")
    print(f"Estructura utilizada: {metrics.data_structure}")
    print()
    print(f"Nodos generados:            {metrics.generated_nodes}")
    depth: int | str = metrics.depth if metrics.depth >= 0 else "N/A"
    print(f"Profundidad:                {depth}")
    print(f"Cantidad de pasos:          {steps}")
    print(f"Estados en el camino:       {steps + 1}")
    print(f"Tiempo de ejecucion:        {metrics.time:.6f} s")
    print(f"Memoria utilizada:          {metrics.memory_mib():.6f} MiB")
    print(
        f"  ({metrics.generated_nodes} nodos x {Metrics.bytes_per_node} bytes = {metrics.memory_bytes()} bytes)"
    )
    print()
    print(f"Complejidad en tiempo (nodos expandidos): {metrics.expanded_nodes}")
    print(f"Complejidad en espacio (nodos en memoria): {metrics.space()}")


if __name__ == "__main__":
    main()
