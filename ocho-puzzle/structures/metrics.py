from __future__ import annotations

import sys
from typing import Any

from structures.node import Node


class Metrics:
    bytes_per_node: int = 0

    def __init__(self) -> None:
        self.algorithm: str = "A*"
        self.data_structure: str = ""
        self.expanded_nodes: int = 0
        self.generated_nodes: int = 0
        self.max_frontier: int = 0
        self.visited: int = 0
        self.time: float = 0.0
        self.depth: int = -1
        if Metrics.bytes_per_node == 0:
            Metrics.bytes_per_node = self._calculate_node_size()

    @staticmethod
    def _calculate_node_size() -> int:
        temp_node: Node = Node((0, 0))
        base_size: int = sys.getsizeof(temp_node)
        dict_size: int = sys.getsizeof(temp_node.__dict__)
        state_size: int = sys.getsizeof(temp_node.state)
        children_size: int = sys.getsizeof(temp_node.children)
        action_size: int = sys.getsizeof(temp_node.action)
        return base_size + dict_size + state_size + children_size + action_size

    def set_algorithm(self, name: str) -> None:
        self.algorithm = name

    def set_data_structure(self, name: str) -> None:
        self.data_structure = name

    def expand(self) -> None:
        self.expanded_nodes += 1

    def register_generated(self, count: int = 1) -> None:
        self.generated_nodes += count

    def update_frontier(self, frontier: list[Any]) -> None:
        self.max_frontier = max(self.max_frontier, len(frontier))

    def update_visited(self, visited: dict[Any, Any] | set[Any]) -> None:
        self.visited = len(visited)

    def space(self) -> int:
        return self.max_frontier + self.visited

    def memory_bytes(self) -> int:
        return self.generated_nodes * Metrics.bytes_per_node

    def memory_mib(self) -> float:
        return self.memory_bytes() / (1024 * 1024)

    def finish(self, elapsed: float, depth: int) -> None:
        self.time = elapsed
        self.depth = depth

    def copy(self) -> Metrics:
        clone: Metrics = Metrics()
        clone.algorithm = self.algorithm
        clone.data_structure = self.data_structure
        clone.expanded_nodes = self.expanded_nodes
        clone.generated_nodes = self.generated_nodes
        clone.max_frontier = self.max_frontier
        clone.visited = self.visited
        clone.time = self.time
        clone.depth = self.depth
        return clone
