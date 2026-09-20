from __future__ import annotations
from os import grantpt

from estrategias.base import Strategy
from estrategias.metrics import Metrics
from estrategias.node import Node

GRAFO = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": ["H"],
    "E": ["I"],
    "I": ["J"],
    "J": ["K"],
}


class DFSStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[Node] = []

    def search(
        self,
        start: str,
        goal: str,
        max_x: int,
        max_y: int,
    ) -> list[Node]:
        self.metrics = Metrics()
        self.stack = []

        root: Node = Node(start)
        self.stack.append(root)
        self.tree.set_root(root)

        visited: set[str] = set()
        solutions: list[list[Node]] = []

        while self.stack:
            current: Node = self.stack.pop()

            if current.name in visited:
                continue

            if current.is_goal(goal):
                solutions.append(current.path())
                self.metrics.expand()
                continue

            visited.add(current.name)
            self.metrics.update_visited(visited)
            self.metrics.expand()

            children: list[Node] = self._generate_children(current, goal, max_x, max_y)
            current.children = children

            for child in reversed(children):
                if child.name not in visited:
                    self.stack.append(child)

            self.metrics.update_frontier(self.stack)

        if solutions:
            depths: list[int] = [len(c) for c in solutions]
            best: int = min(depths)
            best_index: int = depths.index(best)
            return solutions[best_index]

        return []

    def _search_all(
        self,
        start: str,
        goal: str,
        max_x: int,
        max_y: int,
    ) -> list[list[Node]]:
        temp_metrics: Metrics = Metrics()
        temp_stack: list[Node] = []

        root: Node = Node(start)
        temp_stack.append(root)

        visited: set[str] = set()
        solutions: list[list[Node]] = []

        while temp_stack:
            current: Node = temp_stack.pop()

            if current.name in visited:
                continue

            if current.is_goal(goal):
                solutions.append(current.path())
                temp_metrics.expand()
                continue

            visited.add(current.name)
            temp_metrics.expand()

            children: list[Node] = self._generate_children(current, goal, max_x, max_y)
            current.children = children

            for child in reversed(children):
                if child.name not in visited:
                    temp_stack.append(child)

        return solutions

    def _generate_children(
        self,
        node: Node,
        goal: str,
        max_x: int,
        max_y: int,
    ) -> list[Node]:
        children: list[Node] = []

        try:
            lista = GRAFO[node.name]
        except Exception as e:
            return children

        if len(lista) == 0:
            return children

        for nodo_hijo in lista:
            child: Node = Node(
                name=nodo_hijo,
                parent=node,
                cost=node.cost + 1,
                depth=node.depth + 1,
            )
            children.append(child)

        return children

    def data_structure_used(self) -> str:
        return "Pila"

    def time_complexity(self) -> str:
        return "O(b^m)"

    def space_complexity(self) -> str:
        return "O(b*m)"
