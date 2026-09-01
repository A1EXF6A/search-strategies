from algorithms.genetic import GeneticAlgorithm
from algorithms.local_search import LocalSearch


def main():
    print("== Algoritmo genético ==")
    print("1. Algoritmo genético")
    print("2. Búsqueda local")

    option: str = input("Selecciona un algoritmo: ")

    if option == "1":
        algorithm: GeneticAlgorithm = GeneticAlgorithm()

        algorithm.run()
    elif option == "2":
        local_search: LocalSearch = LocalSearch()

        local_search.run()
    else:
        print("Opción no válida")


if __name__ == "__main__":
    main()
