import random

from estrategias.base import Strategy
from estrategias.bfs_strategy import BFSStrategy
from estrategias.dfs_strategy import DFSStrategy

MAX_X: int = 10
MAX_Y: int = 20

OBJETIVE_POSITION: tuple[int, int] = (1, 10)

strategy: Strategy | None = None


def main():
    print("--- Agente de Búsqueda ---")

    print("\n--- ¿Donde empezará la búsqueda? ---")
    print("1. Ubicación aleatoria")
    print("2. Ubicación específica")

    opcion: str = input("Elige una opcion (1-2): ")

    agent_position: tuple[int, int] = (0, 0)

    if opcion == "1":
        agent_position = (random.randint(1, MAX_X), random.randint(1, MAX_Y))
    elif opcion == "2":
        x: int = int(input(f"Ingresa la coordenada X (0-{MAX_X}): "))
        y: int = int(input(f"Ingresa la coordenada Y (0-{MAX_Y}): "))
        agent_position = (x, y)
    else:
        agent_position = (random.randint(0, MAX_X), random.randint(0, MAX_Y))

    print("\n--- Resumen de la búsqueda ---")

    print("Ubicación del agente: ")
    print("Ubicación del objetivo:  \n")

    print(f": {agent_position}")
    print(f": {OBJETIVE_POSITION}")

    print("\nMapa:")

    print_map(agent_position, OBJETIVE_POSITION)

    print()

    print("\n--- ¿Qué estrategia se utilizará? ---")
    print("1. Búsqueda por anchura")
    print("2. Búsqueda por profundidad")
    print("3. Búsqueda iterativa")
    print("4. Búsqueda bidireccional")

    opcion: str = input("Elige una opcion (1-4): ")

    if opcion == "1":
        strategy = BFSStrategy()
    elif opcion == "2":
        strategy = DFSStrategy()
    elif opcion == "3":
        print("Búsqueda iterativa no implementada.")
    elif opcion == "4":
        print("Búsqueda bidireccional no implementada.")
    else:
        print("Opcion no valida. Intenta de nuevo.")


def print_map(agent_position: tuple[int, int], objective_position: tuple[int, int]):
    print(f"   +{'-' * 39}+")

    agent_in_y: bool = False
    objective_in_y: bool = False

    for i in range(MAX_Y, 0, -1):
        print(f"{str(i).ljust(3)}|", end="")

        agent_in_y = agent_position[1] == i
        objective_in_y = objective_position[1] == i

        for j in range(MAX_X):
            if agent_in_y and (agent_position[0] - 1) == j:
                print("  |", end="")
                continue

            if objective_in_y and (objective_position[0] - 1) == j:
                print("  |", end="")
                continue

            print("   |", end="")

            if j == MAX_X - 1:
                print()

    print(f"   +{'-' * 39}+")

    print("     ", end="")

    for i in range(1, MAX_X + 1):
        print(f"{i}   ", end="")


if __name__ == "__main__":
    main()
