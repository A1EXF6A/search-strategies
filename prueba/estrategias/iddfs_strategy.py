from __future__ import annotations

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


class IDDFSStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.max_depth: int = 50

    def search(
        self,
        start: str,
        goal: str,
        max_x: int,
        max_y: int,
    ) -> list[Node]:
        self.metrics = Metrics()

        for depth_limit in range(self.max_depth + 1):
            root: Node = Node(start)
            self.tree.set_root(root)

            result: list[Node] = self._dfs_limited(
                root, goal, max_x, max_y, depth_limit
            )

            if result:
                return result

        return []

    def _dfs_limited(
        self,
        node: Node,
        goal: str,
        max_x: int,
        max_y: int,
        limit: int,
    ) -> list[Node]:
        self.metrics.expand()

        if node.is_goal(goal):
            return node.path()

        if node.depth >= limit:
            return []

        children: list[Node] = self._generate_children(node, goal, max_x, max_y)
        node.children = children

        for child in children:
            result: list[Node] = self._dfs_limited(child, goal, max_x, max_y, limit)
            if result:
                return result

        return []

    def _search_all(
        self,
        start: str,
        goal: str,
        max_x: int,
        max_y: int,
    ) -> list[list[Node]]:
        for depth_limit in range(self.max_depth + 1):
            root: Node = Node(start)
            solutions: list[list[Node]] = []

            self._dfs_limited_all(root, goal, max_x, max_y, depth_limit, solutions)

            if solutions:
                return solutions

        return []

    def _dfs_limited_all(
        self,
        node: Node,
        goal: str,
        max_x: int,
        max_y: int,
        limit: int,
        solutions: list[list[Node]],
    ) -> None:
        if node.is_goal(goal):
            solutions.append(node.path())
            return

        if node.depth >= limit:
            return

        children: list[Node] = self._generate_children(node, goal, max_x, max_y)
        node.children = children

        for child in children:
            self._dfs_limited_all(child, goal, max_x, max_y, limit, solutions)

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
        return "O(b^d)"

    def space_complexity(self) -> str:
        return "O(b*d)"
