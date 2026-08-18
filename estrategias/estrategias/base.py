from __future__ import annotations

import time
from abc import ABC, abstractmethod

from estrategias.metrics import Metrics
from estrategias.node import Node
from estrategias.tree import Tree


class Strategy(ABC):
    def __init__(self) -> None:
        self.tree: Tree = Tree()
        self.metrics: Metrics = Metrics()

    @abstractmethod
    def data_structure_used(self) -> str:
        pass

    @abstractmethod
    def is_optimal(self) -> bool:
        pass

    @abstractmethod
    def search(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        max_x: int,
        max_y: int,
    ) -> list[Node]:
        pass

    def run(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        max_x: int,
        max_y: int,
    ) -> tuple[list[Node], list[list[Node]], Tree, Metrics]:
        Node.reset_counter()
        start_time: float = time.perf_counter()
        optimal_path: list[Node] = self.search(start, goal, max_x, max_y)
        end_time: float = time.perf_counter()
        self.metrics.execution_time = end_time - start_time
        self.metrics.set_total_nodes(Node._counter)
        all_solutions: list[list[Node]] = self._search_all(start, goal, max_x, max_y)
        return optimal_path, all_solutions, self.tree, self.metrics

    @abstractmethod
    def _search_all(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        max_x: int,
        max_y: int,
    ) -> list[list[Node]]:
        pass
