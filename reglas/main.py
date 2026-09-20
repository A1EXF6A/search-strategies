from itertools import combinations
from pathlib import Path
from typing import cast

import pandas as pd
from rule import ItemSet, Rule

MIN_SUPPORT: float = 0.7

MIN_CONFIDENCE: float = 0.8

RESET: str = "\033[0m"
GREEN: str = "\033[92m"
YELLOW: str = "\033[93m"
RED: str = "\033[91m"
BLUE: str = "\033[94m"


def main() -> None:
    path: Path = Path(__file__).parent / "dataset_apriori_supermercado.csv"

    df: pd.DataFrame = load_csv(path)

    min_coverage: int = int(MIN_SUPPORT * df.shape[0])

    print()
    print(f"Cobertura mínima {MIN_SUPPORT} - {min_coverage} elementos")
    print()

    frequent_by_k: list[list[ItemSet]] = []

    # Phase 1: Filter individual items (k = 1)

    k: int = 1

    frequent_one: list[ItemSet] = filter_individual_items(df, min_coverage)

    frequent_by_k.append(frequent_one)

    print_iteration(k, frequent_one, min_coverage)

    # Phases k = 2, 3, ... until no more candidates are generated

    k = 2

    while True:
        candidates: list[ItemSet] = generate_candidates(frequent_by_k[-1], k)

        if not candidates:
            print(f"{YELLOW}Iteración k = {k}: sin candidatos{RESET}")
            print()
            break

        frequent_k: list[ItemSet] = filter_by_support(candidates, df, min_coverage)

        frequent_by_k.append(frequent_k)

        print_iteration(k, frequent_k, min_coverage)

        k += 1

    # Phase 2: generate association rules A -> B

    print(f"{GREEN}Reglas de asociación generadas:{RESET}")
    print()

    rules: list[Rule] = generate_rules(frequent_by_k, df)

    if not rules:
        print("  (no se generaron reglas que cumplan la confianza mínima)")
        print()
        return

    index: int = 1

    for rule in rules:
        print(f"  {index}. {format_rule(rule)}")
        print(
            f"     > confianza={rule.confidence * 100:.1f}%  soporte={rule.support:>3}  lift={rule.lift:.2f}"
        )
        index += 1

    print()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    df: pd.DataFrame = pd.read_csv(path, usecols=lambda c: c != "Transaccion_ID")

    print(f"Dataset: {path.name}")

    return df


def filter_individual_items(df: pd.DataFrame, min_coverage: int) -> list[ItemSet]:
    passed: list[ItemSet] = []

    for column in df.columns:
        counts: pd.Series = df[column].value_counts()

        sold_count: int = int(cast(float, counts.get(1, 0)))
        not_sold_count: int = int(cast(float, counts.get(0, 0)))

        if sold_count >= min_coverage:
            passed.append(ItemSet(frozenset({(column, True)}), sold_count))

        if not_sold_count >= min_coverage:
            passed.append(ItemSet(frozenset({(column, False)}), not_sold_count))

    return passed


def generate_candidates(prev: list[ItemSet], k: int) -> list[ItemSet]:
    previous_set: list[frozenset[tuple[str, bool]]] = [
        itemset.items for itemset in prev
    ]

    candidates: set[frozenset[tuple[str, bool]]] = set()

    for i, left in enumerate(previous_set):
        for right in previous_set[i + 1 :]:
            candidate: frozenset[tuple[str, bool]] = left | right

            if len(candidate) != k:
                continue

            if has_conflicting_items(candidate):
                continue

            if not all_subsets_frequent(candidate, previous_set):
                continue

            candidates.add(candidate)

    return [ItemSet(items, 0) for items in candidates]


def has_conflicting_items(itemset: frozenset[tuple[str, bool]]) -> bool:
    seen_columns: set[str] = set()

    for column, _ in itemset:
        if column in seen_columns:
            return True

        seen_columns.add(column)

    return False


def all_subsets_frequent(
    itemset: frozenset[tuple[str, bool]],
    previous_set: list[frozenset[tuple[str, bool]]],
) -> bool:
    item_list: list[tuple[str, bool]] = list(itemset)

    for subset in combinations(item_list, len(item_list) - 1):
        if frozenset(subset) not in previous_set:
            return False

    return True


def filter_by_support(
    candidates: list[ItemSet], df: pd.DataFrame, min_coverage: int
) -> list[ItemSet]:
    frequent: list[ItemSet] = []

    for candidate in candidates:
        support: int = count_support(candidate.items, df)

        if support >= min_coverage:
            frequent.append(ItemSet(candidate.items, support))

    return frequent


def count_support(itemset: frozenset[tuple[str, bool]], df: pd.DataFrame) -> int:
    mask: pd.Series = pd.Series(True, index=df.index)

    for column, sold in itemset:
        expected: int = 1 if sold else 0
        mask &= df[column] == expected

    return int(mask.sum())


def generate_rules(frequent_by_k: list[list[ItemSet]], df: pd.DataFrame) -> list[Rule]:
    support_by_itemset: dict[frozenset[tuple[str, bool]], int] = {}

    for group in frequent_by_k:
        for itemset in group:
            support_by_itemset[itemset.items] = itemset.support

    rules: list[Rule] = []

    for group in frequent_by_k:
        for itemset in group:
            if len(itemset.items) < 2:
                continue

            item_list: list[tuple[str, bool]] = list(itemset.items)

            for size in range(1, len(item_list)):
                for subset in combinations(item_list, size):
                    antecedent: frozenset[tuple[str, bool]] = frozenset(subset)
                    consequent: frozenset[tuple[str, bool]] = itemset.items - antecedent

                    if not consequent:
                        continue

                    antecedent_support: int = support_by_itemset.get(antecedent, 0)

                    if antecedent_support == 0:
                        continue

                    confidence: float = itemset.support / antecedent_support

                    count_consecuent: int = count_support(consequent, df)

                    lift: float = confidence / (count_consecuent / df.shape[0])

                    if confidence >= MIN_CONFIDENCE:
                        rules.append(
                            Rule(
                                antecedent,
                                consequent,
                                itemset.support,
                                confidence,
                                lift,
                            )
                        )

    return rules


def print_iteration(k: int, itemsets: list[ItemSet], min_coverage: int) -> None:
    print(f"{BLUE}Iteración k = {k}{RESET}")

    if not itemsets:
        print("  (no hay itemsets frecuentes)")
        print()
        return

    for itemset in itemsets:
        print(f"  {format_itemset(itemset.items)}  soporte {itemset.support:>3}")

    print()


def format_itemset(itemset: frozenset[tuple[str, bool]]) -> str:
    parts: list[str] = []

    for column, sold in sorted(itemset):
        parts.append(column if sold else f"~{column}")

    return " ^ ".join(parts).ljust(25)


def format_rule(rule: Rule) -> str:
    antecedent_parts: list[str] = []

    for column, sold in sorted(rule.antecedent):
        antecedent_parts.append(column if sold else f"~{column}")

    antecedent_str: str = " AND ".join(antecedent_parts)

    consecuent_parts: list[str] = []

    for column, sold in sorted(rule.consequent):
        consecuent_parts.append(column if sold else f"~{column}")

    consecuent_str: str = " AND ".join(consecuent_parts)

    return f"SI {YELLOW}[{antecedent_str}]{RESET} ENTONCES {YELLOW}[{consecuent_str}]{RESET}"


if __name__ == "__main__":
    main()
