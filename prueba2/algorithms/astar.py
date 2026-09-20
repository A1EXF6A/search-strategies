from __future__ import annotations

import heapq
import time
from collections.abc import Callable

from states.state import PuzzleState, heuristic
from structures.metrics import Metrics
from structures.nodea import NodeA


def solve_a_star(
    initial_state: str,
    goal_state: str,
    heuristic: Callable[[str], int] = heuristic,
) -> tuple[NodeA | None, NodeA, Metrics]:
    NodeA.reset_counter()
    root: NodeA = NodeA(initial_state, h=heuristic(initial_state))
    goal: NodeA = NodeA(goal_state, h=heuristic(goal_state))

    frontier: list[NodeA] = []
    heapq.heappush(frontier, root)

    visited: dict[NodeA, int] = {root: 0}
    metrics: Metrics = Metrics()
    metrics.set_algorithm("A*")
    metrics.set_data_structure("Cola de prioridad")
    metrics.register_generated()
    metrics.update_frontier(frontier)
    metrics.update_visited(visited)

    t0: float = time.perf_counter()

    while frontier:
        current: NodeA = heapq.heappop(frontier)
        metrics.update_frontier(frontier)

        state: NodeA = current  # type: ignore[assignment]
        if state.is_goal(goal):
            metrics.finish(time.perf_counter() - t0, current.depth)
            return current, root, metrics

        metrics.expand()

        for successor in state.successors():
            new_g: int = current.g + 1
            if successor in visited and visited[successor] <= new_g:
                continue

            visited[successor] = new_g
            metrics.update_visited(visited)
            h_val: int = heuristic(successor.data)
            child: NodeA = NodeA(
                successor.data,
                parent=current,
                g=new_g,
                h=h_val,
            )
            current.children.append(child)
            heapq.heappush(frontier, child)
            metrics.register_generated()
            metrics.update_frontier(frontier)

    metrics.finish(time.perf_counter() - t0, -1)
    return None, root, metrics
