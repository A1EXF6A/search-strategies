from __future__ import annotations

from estrategias.node import Node


class Tree:
    RESET: str = "\033[0m"
    GREEN: str = "\033[92m"
    YELLOW: str = "\033[93m"

    def __init__(self, root: Node | None = None):
        self.root: Node | None = root

    def set_root(self, root: Node) -> None:
        self.root = root

    def path_from(self, node: Node) -> list[Node]:
        if node is None:
            return []
        return node.path()

    def print(
        self,
        optimal_path: list[Node] | None = None,
        goals: list[Node] | None = None,
    ) -> None:
        if self.root is None:
            return
        optimal_path = optimal_path or []
        goals = goals or []
        self._print_tree_with_path(
            self.root,
            optimal_path,
            goals,
        )

    def _print_tree_with_path(
        self,
        root: Node,
        optimal_path: list[Node],
        goals: list[Node],
        prefix: str = "",
        is_last: bool = True,
    ) -> None:
        connector: str = "--- " if is_last else "+-- "
        text: str = (
            f"({root.position[0]},{root.position[1]})"
            + f" Depth:{root.depth}"
            + f" Cost:{root.cost}"
        )

        if root in optimal_path:
            print(prefix + connector + self.GREEN + text + self.RESET)
        elif root in goals:
            print(prefix + connector + self.YELLOW + text + self.RESET)
        else:
            print(prefix + connector + text)

        prefix_children: str = prefix + ("    " if is_last else "|   ")
        children: list[Node] = root.children or []
        for i, child in enumerate(children):
            is_last_child: bool = i == len(children) - 1
            self._print_tree_with_path(
                child,
                optimal_path,
                goals,
                prefix_children,
                is_last_child,
            )
