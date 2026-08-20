from __future__ import annotations

from collections import deque

from estrategias.base import Strategy
from estrategias.metrics import Metrics
from estrategias.node import Node

MOVEMENTS: list[tuple[str, int, int]] = [
    ("UP", 0, 1),
    ("DOWN", 0, -1),
    ("LEFT", -1, 0),
    ("RIGHT", 1, 0),
]


class BidirectionalStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.forward_queue: deque[Node] = deque()
        self.backward_queue: deque[Node] = deque()

    def search(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        max_x: int,
        max_y: int,
    ) -> list[Node]:
        self.metrics = Metrics()
        self.forward_queue = deque()
        self.backward_queue = deque()

        forward_root: Node = Node(start)
        backward_root: Node = Node(goal)

        self.forward_queue.append(forward_root)
        self.backward_queue.append(backward_root)
        self.tree.set_root(forward_root)

        forward_visited: dict[tuple[int, int], Node] = {}
        backward_visited: dict[tuple[int, int], Node] = {}

        while self.forward_queue or self.backward_queue:
            meeting_point: tuple[Node, Node] | None = self._expand_forward(
                forward_visited, backward_visited, max_x, max_y
            )
            if meeting_point:
                return self._build_path(meeting_point[0], meeting_point[1])

            meeting_point = self._expand_backward(
                forward_visited, backward_visited, max_x, max_y
            )
            if meeting_point:
                return self._build_path(meeting_point[0], meeting_point[1])

        return []

    def _expand_forward(
        self,
        forward_visited: dict[tuple[int, int], Node],
        backward_visited: dict[tuple[int, int], Node],
        max_x: int,
        max_y: int,
    ) -> tuple[Node, Node] | None:
        if not self.forward_queue:
            return None

        current: Node = self.forward_queue.popleft()

        if current.position in forward_visited:
            return None

        forward_visited[current.position] = current
        self.metrics.expand()
        self.metrics.update_frontier(self.forward_queue)

        if current.position in backward_visited:
            return (current, backward_visited[current.position])

        children: list[Node] = self._generate_children(current, max_x, max_y)
        current.children = children
        self.metrics.generate(len(children))

        for child in children:
            if child.position not in forward_visited:
                self.forward_queue.append(child)

        return None

    def _expand_backward(
        self,
        forward_visited: dict[tuple[int, int], Node],
        backward_visited: dict[tuple[int, int], Node],
        max_x: int,
        max_y: int,
    ) -> tuple[Node, Node] | None:
        if not self.backward_queue:
            return None

        current: Node = self.backward_queue.popleft()

        if current.position in backward_visited:
            return None

        backward_visited[current.position] = current
        self.metrics.expand()
        self.metrics.update_frontier(self.backward_queue)

        if current.position in forward_visited:
            return (forward_visited[current.position], current)

        children: list[Node] = self._generate_children_backward(current, max_x, max_y)
        self.metrics.generate(len(children))

        for child in children:
            if child.position not in backward_visited:
                self.backward_queue.append(child)

        return None

    def _build_path(self, forward_node: Node, backward_node: Node) -> list[Node]:
        forward_path: list[Node] = forward_node.path()

        current: Node | None = backward_node
        backward_path: list[Node] = []
        while current is not None:
            backward_path.append(current)
            current = current.parent

        if (
            backward_path
            and forward_path
            and backward_path[0].position == forward_path[-1].position
        ):
            backward_path = backward_path[1:]

        combined: list[Node] = forward_path + backward_path

        for i, node in enumerate(combined):
            node.depth = i
            node.cost = i
            if i > 0:
                node.action = self._find_action(combined[i - 1].position, node.position)

        return combined

    def _find_action(self, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> str:
        dx: int = to_pos[0] - from_pos[0]
        dy: int = to_pos[1] - from_pos[1]
        for action, mdx, mdy in MOVEMENTS:
            if mdx == dx and mdy == dy:
                return action
        return "UNKNOWN"

    def _search_all(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        max_x: int,
        max_y: int,
    ) -> list[list[Node]]:
        path: list[Node] = self.search(start, goal, max_x, max_y)
        return [path] if path else []

    def _generate_children(
        self,
        node: Node,
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

    def _generate_children_backward(
        self,
        node: Node,
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
        return "Dos colas"

    def time_complexity(self) -> str:
        return "O(b^(d/2))"

    def space_complexity(self) -> str:
        return "O(b^(d/2))"
