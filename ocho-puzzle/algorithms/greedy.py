from __future__ import annotations

import heapq
import itertools
import time
from collections.abc import Callable

from states.state import PuzzleState, manhattan_heuristic
from structures.metrics import Metrics
from structures.node import Node


def solve_greedy(
    initial_state: PuzzleState,
    heuristic: Callable[[PuzzleState], int] = manhattan_heuristic,
) -> tuple[Node | None, Node, Metrics]:
    """Greedy best-first search: frontier prioritized by h(n) only.

    Unlike A* (f = g + h), this evaluates solely the heuristic,
    so the solution found is not guaranteed to be optimal.
    """
    Node.reset_counter()
    root: Node = Node(initial_state, g=0, h=heuristic(initial_state))

    counter: itertools.count[int] = itertools.count()
    frontier: list[tuple[int, int, Node]] = []
    heapq.heappush(frontier, (root.h, next(counter), root))

    visited: set[tuple[tuple[int, ...], ...]] = {initial_state.board}
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

        state: PuzzleState = current.state  # type: ignore[assignment]
        if state.is_goal():
            metrics.finish(time.perf_counter() - t0, current.depth)
            return current, root, metrics

        metrics.expand()

        for action, successor in state.successors():
            if successor.board in visited:
                continue
            visited.add(successor.board)
            metrics.update_visited(visited)
            child: Node = Node(
                successor,
                parent=current,
                action=action,
                g=current.g + 1,
                h=heuristic(successor),
            )
            current.children.append(child)
            heapq.heappush(frontier, (child.h, next(counter), child))
            metrics.register_generated()
            metrics.update_frontier(frontier)

    metrics.finish(time.perf_counter() - t0, -1)
    return None, root, metrics
