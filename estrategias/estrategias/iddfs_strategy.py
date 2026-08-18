from __future__ import annotations

from estrategias.base import Strategy
from estrategias.metrics import Metrics
from estrategias.node import Node

MOVEMENTS: list[tuple[str, int, int]] = [
    ("UP", 0, 1),
    ("DOWN", 0, -1),
    ("LEFT", -1, 0),
    ("RIGHT", 1, 0),
]


class IDDFSStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.max_depth: int = 50

    def search(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
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
        goal: tuple[int, int],
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
        start: tuple[int, int],
        goal: tuple[int, int],
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
        goal: tuple[int, int],
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
        goal: tuple[int, int],
        max_x: int,
        max_y: int,
    ) -> list[Node]:
        children: list[Node] = []
        for action, dx, dy in MOVEMENTS:
            new_x: int = node.position[0] + dx
            new_y: int = node.position[1] + dy
            new_position: tuple[int, int] = (new_x, new_y)

            if 0 <= new_x <= max_x and 0 <= new_y <= max_y:
                child: Node = Node(
                    position=new_position,
                    parent=node,
                    action=action,
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
