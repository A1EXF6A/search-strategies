from __future__ import annotations

from typing import Any

from states.state import PESOS

GRAFO = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": ["H"],
    "E": ["I"],
    "I": ["J"],
    "J": ["K"],
}


class Node:
    _counter: int = 0

    def __init__(
        self,
        data: str,
        parent: Node | None = None,
        h: int = 0,
    ) -> None:
        Node._counter += 1
        self.id: int = Node._counter
        self.data: Any = data
        self.parent: Node | None = parent
        self.children: list[Node] = []
        self.depth: int = 0 if parent is None else parent.depth + 1
        self.h: int = h
        self.f: int = h

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

    def is_goal(self, other: Node) -> bool:
        return self.data == other.data

    def successors(self) -> list[Node]:
        children: list[Node] = []

        try:
            lista = GRAFO[self.data]
        except Exception as e:
            return children

        if len(lista) == 0:
            return children

        for nodo_hijo in lista:
            child: Node = Node(data=nodo_hijo, parent=self, h=PESOS[nodo_hijo])
            children.append(child)

        return children
