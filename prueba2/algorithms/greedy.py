from __future__ import annotations

import heapq
import itertools
import time
from collections.abc import Callable

from states.state import heuristic
from structures.metrics import Metrics
from structures.node import Node


def solve_greedy(
    initial_state: str,
    goal_state: str,
    heuristic: Callable[[str], int] = heuristic,
) -> tuple[Node | None, Node, Metrics]:
    """Greedy best-first search: frontier prioritized by h(n) only.

    Unlike A* (f = g + h), this evaluates solely the heuristic,
    so the solution found is not guaranteed to be optimal.
    """
    Node.reset_counter()
    root: Node = Node(initial_state, h=heuristic(initial_state))

    goal: Node = Node(goal_state, h=heuristic(goal_state))

    counter: itertools.count[int] = itertools.count()
    frontier: list[tuple[int, int, Node]] = []
    heapq.heappush(frontier, (root.h, next(counter), root))

    visited: set[Node] = {root}
    metrics: Metrics = Metrics()
    metrics.set_algorithm("Greedy")
    metrics.set_data_structure("Cola de prioridad")
    metrics.register_generated()
    metrics.update_frontier(frontier)
    metrics.update_visited(visited)

    t0: float = time.perf_counter()

    while frontier:
        _, _, current = heapq.heappop(frontier)
        metrics.update_frontier(frontier)

        state: Node = current  # type: ignore[assignment]
        if state.is_goal(goal):
            metrics.finish(time.perf_counter() - t0, current.depth)
            return current, root, metrics

        metrics.expand()

        for successor in state.successors():
            if successor in visited:
                continue
            visited.add(successor)
            metrics.update_visited(visited)

            child: Node = Node(
                successor.data,
                parent=current,
                h=heuristic(successor.data),
            )
            current.children.append(child)
            heapq.heappush(frontier, (child.h, next(counter), child))
            metrics.register_generated()
            metrics.update_frontier(frontier)

    metrics.finish(time.perf_counter() - t0, -1)
    return None, root, metrics
