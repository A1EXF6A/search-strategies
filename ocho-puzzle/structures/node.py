from __future__ import annotations

from typing import Any


class Node:
    _counter: int = 0

    def __init__(
        self,
        data: Any,
        parent: Node | None = None,
        action: str | None = None,
        g: int = 0,
        h: int = 0,
    ) -> None:
        Node._counter += 1
        self.id: int = Node._counter
        self.data: Any = data
        self.state: Any = data
        self.parent: Node | None = parent
        self.children: list[Node] = []
        self.depth: int = 0 if parent is None else parent.depth + 1
        self.action: str | None = action
        self.g: int = g
        self.h: int = h
        self.f: int = g + h

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

    def __lt__(self, other: Node) -> bool:
        return self.f < other.f
