from __future__ import annotations


class Node:
    _counter: int = 0

    _name: str

    def __init__(
        self,
        name: str,
        parent: Node | None = None,
        cost: int = 0,
        depth: int = 0,
    ):
        Node._counter += 1
        self.name = name
        self.id: int = Node._counter
        self.parent: Node | None = parent
        self.children: list[Node] = []
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

    def is_goal(self, goal: str) -> bool:
        return self.name == goal

    def __repr__(self) -> str:
        return (
            f"Node(id={self.id}, name={self.name}, "
            f"depth={self.depth}, cost={self.cost})"
        )
