from __future__ import annotations

from collections import deque

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


class BFSStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.queue: deque[Node] = deque()

    def search(
        self,
        start: str,
        goal: str,
        max_x: int,
        max_y: int,
    ) -> list[Node]:
        self.metrics = Metrics()
        self.queue = deque()

        root: Node = Node(start)
        self.queue.append(root)
        self.tree.set_root(root)

        visited: set[str] = set()
        solutions: list[list[Node]] = []
        found_depth: int = -1

        while self.queue:
            self.metrics.update_frontier(self.queue)
            current: Node = self.queue.popleft()

            if current.name in visited:
                continue

            if found_depth != -1 and current.depth > found_depth:
                break

            if current.is_goal(goal):
                solutions.append(current.path())
                found_depth = current.depth
                self.metrics.expand()
                continue

            visited.add(current.name)
            self.metrics.update_visited(visited)
            self.metrics.expand()

            children: list[Node] = self._generate_children(current, goal, max_x, max_y)
            current.children = children

            for child in children:
                if child.name not in visited:
                    self.queue.append(child)

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
        temp_queue: deque[Node] = deque()

        root: Node = Node(start)
        temp_queue.append(root)

        visited: set[str] = set()
        solutions: list[list[Node]] = []
        found_depth: int = -1

        while temp_queue:
            current: Node = temp_queue.popleft()

            if current.name in visited:
                continue

            if found_depth != -1 and current.depth > found_depth:
                break

            if current.is_goal(goal):
                solutions.append(current.path())
                found_depth = current.depth
                temp_metrics.expand()
                continue

            visited.add(current.name)
            temp_metrics.expand()

            children: list[Node] = self._generate_children(current, goal, max_x, max_y)
            current.children = children

            for child in children:
                if child.name not in visited:
                    temp_queue.append(child)

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
        return "Cola"

    def time_complexity(self) -> str:
        return "O(b^d)"

    def space_complexity(self) -> str:
        return "O(b^d)"
