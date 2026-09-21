from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pandas as pd
from structures.rule import Condition, PrismRule

CLASS_COLUMN: str = "Exito_Proyecto"

ID_COLUMN: str = "Proyecto_ID"

MIN_CONFIDENCE: float = 0.85

RESET: str = "\033[0m"
GREEN: str = "\033[92m"
YELLOW: str = "\033[93m"
RED: str = "\033[91m"
BLUE: str = "\033[94m"


def main() -> None:
    path: Path = Path(__file__).parent / "dataset_prism_proyectos_software.csv"

    # Phase 1: Load the dataset and list the target class values

    df: pd.DataFrame = load_csv(path)

    class_values: list[str] = sorted(
        cast(list[str], df[CLASS_COLUMN].unique().tolist())
    )

    print(f"Clase objetivo: {CLASS_COLUMN} -> {class_values}")
    print()

    # Phase 2: Generate PRISM rules for each class value

    rules: list[PrismRule] = []

    for class_value in class_values:
        rules.extend(generate_rules_for_class(df, class_value))

    # print(f"{GREEN}Reglas PRISM generadas:{RESET}")
    # print()
    #
    # for index, rule in enumerate(rules, 1):
    #     marker: str = (
    #         f"{RED} *{RESET}" if rule.confidence >= MIN_CONFIDENCE else ""
    #     )
    #
    #     print(f"  {index}. {format_rule(rule)}{marker}")
    #     print(
    #         f"     > confianza={rule.confidence * 100:.1f}%  cobertura={rule.coverage:>3}"
    #     )
    #
    # print()
    # print(
    #     f"{YELLOW}* reglas que cumplen la confianza mínima "
    #     f"{MIN_CONFIDENCE:.0%}{RESET}"
    # )
    # print()

    # Phase 3: Summarize the rules generated per class

    print(f"{GREEN}Reglas definidas:{RESET}")
    print()

    for class_value in class_values:
        count: int = sum(1 for rule in rules if rule.class_value == class_value)
        print(f"  {CLASS_COLUMN} = {class_value}: {count} reglas")

    print()

    # Phase 4: Report the most relevant rules

    report_best_rules(df, rules)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    df: pd.DataFrame = pd.read_csv(path, usecols=lambda c: c != ID_COLUMN)

    print(f"Dataset: {path.name}")

    return df


def generate_rules_for_class(df: pd.DataFrame, class_value: str) -> list[PrismRule]:
    remaining: pd.DataFrame = df
    rules: list[PrismRule] = []
    rule_index: int = 1

    print(f"{BLUE}Clase objetivo: {class_value}{RESET}")
    print()

    # Phase A: Repeat while there are still uncovered examples of the class

    while count_examples(remaining, class_value) > 0:
        conditions: list[Condition] = []
        working: pd.DataFrame = remaining

        # print(f"  Construyendo regla {rule_index}")
        # print()

        # Phase B: Grow the rule adding the best condition at each step

        while True:
            choice: tuple[Condition, float, int] | None = best_condition(
                working, class_value, conditions
            )

            if choice is None:
                break

            condition, confidence, total = choice

            conditions.append(condition)

            working = cast(pd.DataFrame, working[matches(working, conditions)])

            # print(
            #     f"    + {condition.column} = {condition.value}  "
            #     f"confianza {confidence * 100:.1f}%  cobertura {total}"
            # )

            # Stop when every covered example belongs to the class

            if bool(cast(pd.Series, working[CLASS_COLUMN]).eq(class_value).all()):
                break

        if not conditions:
            break

        # Phase C: Evaluate the finished rule over the whole dataset

        coverage: int = count_matches(df, conditions)
        positives: int = count_positives(df, conditions, class_value)
        confidence_final: float = positives / coverage if coverage > 0 else 0.0

        rules.append(
            PrismRule(tuple(conditions), class_value, coverage, confidence_final)
        )

        # print(
        #     f"    Regla final: {format_rule_conditions(conditions, class_value)}  "
        #     f"cobertura {coverage}  confianza {confidence_final * 100:.1f}%"
        # )
        # print()

        # Phase D: Remove the covered examples for the next rule to take over

        remaining = cast(pd.DataFrame, remaining[~matches(remaining, conditions)])

        rule_index += 1

    return rules


def report_best_rules(df: pd.DataFrame, rules: list[PrismRule]) -> None:
    qualified: list[PrismRule] = [
        rule for rule in rules if rule.confidence >= MIN_CONFIDENCE
    ]

    print(f"{GREEN}Mejores reglas con confianza >= {MIN_CONFIDENCE:.0%}:{RESET}")
    print()

    if not qualified:
        print("  (no se encontraron reglas que cumplan la confianza mínima)")
        print()
        return

    # Select the best rule per class (highest confidence, then highest coverage)

    ranking: list[PrismRule] = sorted(
        qualified, key=lambda rule: (rule.confidence, rule.coverage), reverse=True
    )

    best_by_class: dict[str, PrismRule] = {}

    for rule in ranking:
        if rule.class_value not in best_by_class:
            best_by_class[rule.class_value] = rule

    for class_value, rule in best_by_class.items():
        print(f"  {BLUE}Mejor regla para {CLASS_COLUMN} = {class_value}:{RESET}")
        print(f"    {format_rule(rule)}")
        print(
            f"    > confianza={rule.confidence * 100:.1f}%  "
            f"cobertura={rule.coverage} ({rule.coverage / df.shape[0] * 100:.1f}% del dataset)"
        )
        print()

    print("  Ranking de reglas (mayor confianza, luego mayor cobertura):")
    print()

    total_rows: int = df.shape[0]

    for index, rule in enumerate(ranking, 1):
        antecedent: str = " AND ".join(
            f"{condition.column} = {condition.value}" for condition in rule.conditions
        )

        support: int = count_positives(df, list(rule.conditions), rule.class_value)

        class_probability: float = count_examples(df, rule.class_value) / total_rows

        lift: float = rule.confidence / class_probability

        print(
            f"    {index:>2}. {YELLOW}[{antecedent}]{RESET} -> "
            f"{CLASS_COLUMN} = {rule.class_value}  "
            f"(confianza {rule.confidence * 100:.1f}%, cobertura {rule.coverage}, "
            f"soporte {support}, lift {lift:.2f})"
        )

    print()


def count_examples(df: pd.DataFrame, class_value: str) -> int:
    return int((df[CLASS_COLUMN] == class_value).sum())


def count_matches(df: pd.DataFrame, conditions: list[Condition]) -> int:
    return int(matches(df, conditions).sum())


def count_positives(
    df: pd.DataFrame, conditions: list[Condition], class_value: str
) -> int:
    mask: pd.Series = matches(df, conditions) & (df[CLASS_COLUMN] == class_value)

    return int(mask.sum())


def matches(df: pd.DataFrame, conditions: list[Condition]) -> pd.Series:
    mask: pd.Series = pd.Series(True, index=df.index)

    for condition in conditions:
        mask &= df[condition.column] == condition.value

    return mask


def best_condition(
    df: pd.DataFrame, class_value: str, used: list[Condition]
) -> tuple[Condition, float, int] | None:
    # Pick the feature-and-value that covers the class with the highest confidence

    used_columns: set[str] = {condition.column for condition in used}

    best: tuple[Condition, float, int] | None = None

    for column in df.columns:
        if column == CLASS_COLUMN or column in used_columns:
            continue

        for value in df[column].unique():
            mask_value: pd.Series = df[column] == value
            total: int = int(mask_value.sum())

            if total == 0:
                continue

            positives: int = int((mask_value & (df[CLASS_COLUMN] == class_value)).sum())

            confidence: float = positives / total

            if (
                best is None
                or confidence > best[1]
                or (confidence == best[1] and total > best[2])
            ):
                best = (Condition(column, value), confidence, total)

    return best


def format_rule(rule: PrismRule) -> str:
    return format_rule_conditions(rule.conditions, rule.class_value)


def format_rule_conditions(conditions: Iterable[Condition], class_value: str) -> str:
    antecedent: str = " AND ".join(
        f"{condition.column} = {condition.value}" for condition in conditions
    )

    return (
        f"SI {YELLOW}[{antecedent}]{RESET} "
        f"ENTONCES {YELLOW}[{CLASS_COLUMN} = {class_value}]{RESET}"
    )


if __name__ == "__main__":
    main()

