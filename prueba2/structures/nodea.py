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


class NodeA:
    _counter: int = 0

    def __init__(
        self,
        data: str,
        parent: NodeA | None = None,
        g: int = 0,
        h: int = 0,
    ) -> None:
        NodeA._counter += 1
        self.id: int = NodeA._counter
        self.data: Any = data
        self.parent: NodeA | None = parent
        self.children: list[NodeA] = []
        self.depth: int = 0 if parent is None else parent.depth + 1
        self.g: int = h
        self.h: int = h
        self.f: int = g + h

    @classmethod
    def reset_counter(cls) -> None:
        cls._counter = 0

    def path(self) -> list[NodeA]:
        route: list[NodeA] = []
        current: NodeA | None = self
        while current is not None:
            route.append(current)
            current = current.parent
        route.reverse()
        return route

    def __lt__(self, other: NodeA) -> bool:
        return self.f < other.f

    def is_goal(self, other: NodeA) -> bool:
        return self.data == other.data

    def successors(self) -> list[NodeA]:
        children: list[NodeA] = []

        try:
            lista = GRAFO[self.data]
        except Exception as e:
            return children

        if len(lista) == 0:
            return children

        for nodo_hijo in lista:
            child: NodeA = NodeA(data=nodo_hijo, parent=self, h=PESOS[nodo_hijo])
            children.append(child)

        return children
