from __future__ import annotations


class Node:
    _counter: int = 0

    def __init__(
        self,
        position: tuple[int, int],
        parent: Node | None = None,
        action: str | None = None,
        cost: int = 0,
        depth: int = 0,
    ):
        Node._counter += 1
        self.id: int = Node._counter
        self.position: tuple[int, int] = position
        self.parent: Node | None = parent
        self.children: list[Node] = []
        self.action: str | None = action
        self.cost: int = cost
        self.depth: int = depth

    @classmethod
    def reset_counter(cls) -> None:
        cls._counter = 0

    def path(self) -> list[Node]:
        route: list[Node] = []
        current: Node | None = self
        while current is not None:
            route.append(current)
            current = current.parent
        route.reverse()
        return route

    def is_goal(self, goal: tuple[int, int]) -> bool:
        return self.position == goal

    def __repr__(self) -> str:
        return (
            f"Node(id={self.id}, position={self.position}, "
            f"depth={self.depth}, cost={self.cost})"
        )
