from __future__ import annotations

from structures.node import Node


class Tree:
    RESET: str = "\033[0m"
    GREEN: str = "\033[92m"
    YELLOW: str = "\033[93m"

    def __init__(self, root: Node | None = None) -> None:
        self.root: Node | None = root

    def set_root(self, root: Node) -> None:
        self.root = root

    def path_from(self, node: Node | None) -> list[Node]:
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
        self._print_tree(self.root, optimal_path, goals)

    def _print_tree(
        self,
        root: Node,
        optimal_path: list[Node],
        goals: list[Node],
        prefix: str = "",
        is_last: bool = True,
    ) -> None:
        connector: str = "└── " if is_last else "├── "
        text: str = root.data.to_string() + "  Nivel:" + str(root.depth)

        if root in goals:
            print(prefix + connector + self.YELLOW + text + " ✔" + self.RESET)
        elif root in optimal_path:
            print(prefix + connector + self.GREEN + text + self.RESET)
        else:
            print(prefix + connector + text)

        child_prefix: str = prefix + ("    " if is_last else "│   ")
        children: list[Node] = root.children or []
        for i, child in enumerate(children):
            self._print_tree(
                child,
                optimal_path,
                goals,
                child_prefix,
                i == len(children) - 1,
            )

    def print_limited(
        self,
        optimal_path: list[Node] | None = None,
        goals: list[Node] | None = None,
        max_depth: int = 4,
    ) -> None:
        if self.root is None:
            return
        self._print_limited(
            self.root,
            optimal_path or [],
            goals or [],
            "",
            True,
            max_depth,
        )

    def _print_limited(
        self,
        root: Node,
        optimal_path: list[Node],
        goals: list[Node],
        prefix: str,
        is_last: bool,
        max_depth: int,
    ) -> None:
        if root.depth > max_depth and root not in optimal_path and root not in goals:
            return

        connector: str = "└── " if is_last else "├── "
        text: str = (
            root.data.to_string()
            + f"  f={root.f}(g={root.g}+h={root.h})"
            + "  Nivel:" + str(root.depth)
        )

        if root in goals:
            print(prefix + connector + self.YELLOW + text + " ✔" + self.RESET)
        elif root in optimal_path:
            print(prefix + connector + self.GREEN + text + self.RESET)
        else:
            print(prefix + connector + text)

        child_prefix: str = prefix + ("    " if is_last else "│   ")
        children: list[Node] = root.children or []
        if not children:
            return

        if root.depth < max_depth:
            for i, child in enumerate(children):
                self._print_limited(
                    child,
                    optimal_path,
                    goals,
                    child_prefix,
                    i == len(children) - 1,
                    max_depth,
                )
            return

        visible: list[Node] = [c for c in children if c in optimal_path or c in goals]
        hidden_count: int = len(children) - len(visible)
        entries: list[tuple[str, object]] = []
        if hidden_count > 0:
            entries.append(("hidden", hidden_count))
        entries.extend(("child", c) for c in visible)

        for idx, (kind, value) in enumerate(entries):
            entry_last: bool = idx == len(entries) - 1
            entry_connector: str = "└── " if entry_last else "├── "
            if kind == "hidden":
                print(child_prefix + entry_connector + f"({value} hijos ocultos...)")
                continue
            self._print_limited(
                value,  # type: ignore[arg-type]
                optimal_path,
                goals,
                child_prefix,
                entry_last,
                max_depth,
            )
