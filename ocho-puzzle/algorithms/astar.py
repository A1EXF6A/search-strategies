from __future__ import annotations

import heapq
import time
from collections.abc import Callable

from states.state import PuzzleState, manhattan_heuristic
from structures.metrics import Metrics
from structures.node import Node


def solve_a_star(
    initial_state: PuzzleState,
    heuristic: Callable[[PuzzleState], int] = manhattan_heuristic,
) -> tuple[Node | None, Node, Metrics]:
    Node.reset_counter()
    root: Node = Node(initial_state, g=0, h=heuristic(initial_state))
    frontier: list[Node] = []
    heapq.heappush(frontier, root)

    visited: dict[tuple[tuple[int, ...], ...], int] = {initial_state.board: 0}
    metrics: Metrics = Metrics()
    metrics.set_algorithm("A*")
    metrics.set_data_structure("Cola de prioridad")
    metrics.register_generated()
    metrics.update_frontier(frontier)
    metrics.update_visited(visited)

    t0: float = time.perf_counter()

    while frontier:
        current: Node = heapq.heappop(frontier)
        metrics.update_frontier(frontier)

        state: PuzzleState = current.state  # type: ignore[assignment]
        if state.is_goal():
            metrics.finish(time.perf_counter() - t0, current.depth)
            return current, root, metrics

        metrics.expand()

        for action, successor in state.successors():
            new_g: int = current.g + 1
            if successor.board in visited and visited[successor.board] <= new_g:
                continue
            visited[successor.board] = new_g
            metrics.update_visited(visited)
            h_val: int = heuristic(successor)
            child: Node = Node(
                successor,
                parent=current,
                action=action,
                g=new_g,
                h=h_val,
            )
            current.children.append(child)
            heapq.heappush(frontier, child)
            metrics.register_generated()
            metrics.update_frontier(frontier)

    metrics.finish(time.perf_counter() - t0, -1)
    return None, root, metrics
