from __future__ import annotations

from collections import deque


class Metrics:
    def __init__(self) -> None:
        self.expanded_nodes: int = 0
        self.max_frontier: int = 0
        self.visited: int = 0

    def expand(self) -> None:
        self.expanded_nodes += 1

    def update_frontier(self, frontier: list | deque) -> None:
        if len(frontier) > self.max_frontier:
            self.max_frontier = len(frontier)

    def update_visited(self, visited: set) -> None:
        self.visited = len(visited)

    def space(self) -> int:
        return self.max_frontier + self.visited

    def copy(self) -> Metrics:
        new_metrics: Metrics = Metrics()
        new_metrics.expanded_nodes = self.expanded_nodes
        new_metrics.max_frontier = self.max_frontier
        new_metrics.visited = self.visited
        return new_metrics
