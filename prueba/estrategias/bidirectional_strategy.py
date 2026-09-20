from __future__ import annotations

from collections import deque

from estrategias.base import Strategy
from estrategias.metrics import Metrics
from estrategias.node import Node

GRAFO = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": ["H"],
    "E": ["I"],
    "I": ["J"],
    "J": ["K"],
}


class BidirectionalStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.forward_queue: deque[Node] = deque()
        self.backward_queue: deque[Node] = deque()

    def search(
        self,
        start: str,
        goal: str,
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

        forward_visited: dict[str, Node] = {}
        backward_visited: dict[str, Node] = {}

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
        forward_visited: dict[str, Node],
        backward_visited: dict[str, Node],
        max_x: int,
        max_y: int,
    ) -> tuple[Node, Node] | None:
        if not self.forward_queue:
            return None

        current: Node = self.forward_queue.popleft()

        if current.name in forward_visited:
            return None

        forward_visited[current.name] = current
        self.metrics.expand()
        self.metrics.update_frontier(self.forward_queue)

        if current.name in backward_visited:
            return (current, backward_visited[current.name])

        children: list[Node] = self._generate_children(current, max_x, max_y)
        current.children = children
        self.metrics.generate(len(children))

        for child in children:
            if child.name not in forward_visited:
                self.forward_queue.append(child)

        return None

    def _expand_backward(
        self,
        forward_visited: dict[str, Node],
        backward_visited: dict[str, Node],
        max_x: int,
        max_y: int,
    ) -> tuple[Node, Node] | None:
        if not self.backward_queue:
            return None

        current: Node = self.backward_queue.popleft()

        if current.name in backward_visited:
            return None

        backward_visited[current.name] = current
        self.metrics.expand()
        self.metrics.update_frontier(self.backward_queue)

        if current.name in forward_visited:
            return (forward_visited[current.name], current)

        children: list[Node] = self._generate_children_backward(current, max_x, max_y)
        self.metrics.generate(len(children))

        for child in children:
            if child.name not in backward_visited:
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
            and backward_path[0].name == forward_path[-1].name
        ):
            backward_path = backward_path[1:]

        combined: list[Node] = forward_path + backward_path

        for i, node in enumerate(combined):
            node.depth = i
            node.cost = i

        return combined

    def _search_all(
        self,
        start: str,
        goal: str,
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

        try:
            lista = GRAFO[node.name]
        except Exception as e:
            return children

        if len(lista) == 0:
            return children

        for nodo_hijo in lista:
            child: Node = Node(
                name=nodo_hijo,
                parent=node,
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

        try:
            lista = GRAFO[node.name]
        except Exception as e:
            return children

        if len(lista) == 0:
            return children

        for nodo_hijo in lista:
            child: Node = Node(
                name=nodo_hijo,
                parent=node,
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
