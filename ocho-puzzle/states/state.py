from __future__ import annotations

import random
from collections.abc import Iterable


def _is_solvable(board: tuple[tuple[int, ...], ...]) -> bool:
    values: list[int] = [v for row in board for v in row if v != 0]
    inversions: int = 0
    for i in range(len(values)):
        vi: int = values[i]
        for j in range(i + 1, len(values)):
            if vi > values[j]:
                inversions += 1
    return inversions % 2 == 0


def generate_random_state() -> tuple[tuple[int, ...], ...]:
    while True:
        numbers: list[int] = list(range(9))
        random.shuffle(numbers)
        board: tuple[tuple[int, ...], ...] = tuple(
            tuple(numbers[i * 3 : (i + 1) * 3]) for i in range(3)
        )
        if _is_solvable(board):
            return board


INITIAL_STATE: tuple[tuple[int, ...], ...] = generate_random_state()

GOAL_STATE: tuple[tuple[int, ...], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
)

MOVES: list[tuple[str, int, int]] = [
    ("ARRIBA", -1, 0),
    ("ABAJO", 1, 0),
    ("IZQUIERDA", 0, -1),
    ("DERECHA", 0, 1),
]


class PuzzleState:
    def __init__(self, board: tuple[tuple[int, ...], ...]) -> None:
        self.board: tuple[tuple[int, ...], ...] = board

    def is_goal(self) -> bool:
        return self.board == GOAL_STATE

    def blank_position(self) -> tuple[int, int]:
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return i, j
        raise ValueError("Estado invalido: falta la casilla vacia")

    def successors(self) -> list[tuple[str, PuzzleState]]:
        bi, bj = self.blank_position()
        result: list[tuple[str, PuzzleState]] = []
        for name, di, dj in MOVES:
            ni, nj = bi + di, bj + dj
            if 0 <= ni < 3 and 0 <= nj < 3:
                new_board: list[list[int]] = [list(row) for row in self.board]
                new_board[bi][bj], new_board[ni][nj] = (
                    new_board[ni][nj],
                    new_board[bi][bj],
                )
                successor: PuzzleState = PuzzleState(tuple(tuple(r) for r in new_board))
                result.append((name, successor))
        return result

    def to_string(self) -> str:
        rows: list[str] = []
        for row in self.board:
            rows.append("[" + " ".join("□" if x == 0 else str(x) for x in row) + "]")
        return " ".join(rows)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PuzzleState) and self.board == other.board

    def __hash__(self) -> int:
        return hash(self.board)


GOAL_POS: dict[int, tuple[int, int]] = {
    val: (i, j) for i, row in enumerate(GOAL_STATE) for j, val in enumerate(row)
}


def manhattan_heuristic(state: PuzzleState) -> int:
    dist: int = 0
    for i in range(3):
        for j in range(3):
            val: int = state.board[i][j]
            if val != 0:
                oi, oj = GOAL_POS[val]
                dist += abs(i - oi) + abs(j - oj)
    return dist


def board_from_iterable(rows: Iterable[Iterable[int]]) -> PuzzleState:
    board: tuple[tuple[int, ...], ...] = tuple(tuple(row) for row in rows)
    return PuzzleState(board)
