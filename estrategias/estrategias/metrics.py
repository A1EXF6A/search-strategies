from __future__ import annotations

import sys
from collections import deque

from estrategias.node import Node


class Metrics:
    bytes_per_node: int = 0

    def __init__(self) -> None:
        self.generated_nodes: int = 0
        self.expanded_nodes: int = 0
        self.max_frontier: int = 0
        self.visited: int = 0
        self.execution_time: float = 0.0
        self.total_nodes: int = 0
        if Metrics.bytes_per_node == 0:
            Metrics.bytes_per_node = self._calculate_node_size()

    @staticmethod
    def _calculate_node_size() -> int:
        temp_node: Node = Node((0, 0))
        base_size: int = sys.getsizeof(temp_node)
        dict_size: int = sys.getsizeof(temp_node.__dict__)
        position_size: int = sys.getsizeof(temp_node.position)
        children_size: int = sys.getsizeof(temp_node.children)
        action_size: int = sys.getsizeof(temp_node.action)
        return base_size + dict_size + position_size + children_size + action_size

    def generate(self, count: int = 1) -> None:
        self.generated_nodes += count

    def expand(self) -> None:
        self.expanded_nodes += 1

    def update_frontier(self, frontier: list | deque) -> None:
        self.max_frontier = max(self.max_frontier, len(frontier))

    def update_visited(self, visited: set) -> None:
        self.visited = len(visited)

    def set_generated_nodes(self, count: int) -> None:
        self.generated_nodes = count
        self.total_nodes = count

    def memory_bytes(self) -> int:
        return self.total_nodes * Metrics.bytes_per_node

    def memory_mib(self) -> float:
        return self.memory_bytes() / (1024 * 1024)

    def copy(self) -> Metrics:
        new_metrics: Metrics = Metrics()
        new_metrics.generated_nodes = self.generated_nodes
        new_metrics.expanded_nodes = self.expanded_nodes
        new_metrics.max_frontier = self.max_frontier
        new_metrics.visited = self.visited
        new_metrics.execution_time = self.execution_time
        new_metrics.total_nodes = self.total_nodes
        return new_metrics
