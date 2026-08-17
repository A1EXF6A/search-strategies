from __future__ import annotations

from collections import deque

from estrategias.base import Strategy
from estrategias.metrics import Metrics
from estrategias.node import Node

MOVEMENTS: list[tuple[str, int, int]] = [
    ("UP", -1, 0),
    ("DOWN", 1, 0),
    ("LEFT", 0, -1),
    ("RIGHT", 0, 1),
]


class BFSStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.queue: deque[Node] = deque()

    def search(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        max_x: int,
        max_y: int,
    ) -> list[Node]:
        self.metrics = Metrics()
        self.queue = deque()

        root: Node = Node(start)
        self.queue.append(root)
        self.tree.set_root(root)

        visited: set[tuple[int, int]] = set()
        solutions: list[list[Node]] = []

        while self.queue:
            self.metrics.update_frontier(self.queue)
            current: Node = self.queue.popleft()

            if current.position in visited:
                continue
            visited.add(current.position)
            self.metrics.update_visited(visited)
            self.metrics.expand()

            if current.is_goal(goal):
                solutions.append(current.path())
                continue

            children: list[Node] = self._generate_children(current, goal, max_x, max_y)
            current.children = children

            for child in children:
                if child.position not in visited:
                    self.queue.append(child)

        if solutions:
            depths: list[int] = [len(c) for c in solutions]
            best: int = min(depths)
            best_index: int = depths.index(best)
            return solutions[best_index]

        return []

    def _search_all(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        max_x: int,
        max_y: int,
    ) -> list[list[Node]]:
        temp_metrics: Metrics = Metrics()
        temp_queue: deque[Node] = deque()

        root: Node = Node(start)
        temp_queue.append(root)

        visited: set[tuple[int, int]] = set()
        solutions: list[list[Node]] = []

        while temp_queue:
            current: Node = temp_queue.popleft()

            if current.position in visited:
                continue
            visited.add(current.position)
            temp_metrics.expand()

            if current.is_goal(goal):
                solutions.append(current.path())
                continue

            children: list[Node] = self._generate_children(current, goal, max_x, max_y)
            current.children = children

            for child in children:
                if child.position not in visited:
                    temp_queue.append(child)

        return solutions

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
        return "Cola"

    def is_optimal(self) -> bool:
        return True
