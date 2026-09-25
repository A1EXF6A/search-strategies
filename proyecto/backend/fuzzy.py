import math
import warnings
from typing import cast

import numpy as np
import skfuzzy as fuzz
from numpy.typing import NDArray
from skfuzzy.control import Antecedent, Consequent, ControlSystem, ControlSystemSimulation, Rule
from skfuzzy.control.controlsystem import CrispValueCalculator
from skfuzzy.control.state import StatePerSimulation

from config import (
    DEFAULT_PARAMETERS,
    FUZZY_RULES,
    FUZZY_VARIABLES,
    MISSING_RULES_MESSAGE,
    OUTPUT_MEMBERSHIPS,
    OUTPUT_TERMS,
    RISK_MAX,
    RISK_MIN,
    TERMS,
    UNIVERSE_POINTS,
    VARIABLE_SPECS,
    VariableSpec,
)

FloatArray = NDArray[np.float64]


def _sim_state(property_value: object) -> StatePerSimulation:
    return cast(StatePerSimulation, property_value)


def _sim_float(property_value: object, simulation: object) -> float:
    return float(cast(float, _sim_state(property_value)[simulation]))

warnings.filterwarnings(
    "ignore",
    message=r"Passing more than 2 positional arguments to np\.(maximum|minimum)",
    category=DeprecationWarning,
)


class FuzzySimulation(ControlSystemSimulation):
    def __init__(
        self,
        system: ControlSystem,
        ordered_rules: list[Rule],
        flush_after_run: int,
    ) -> None:
        ControlSystemSimulation.__init__(self, system, flush_after_run=flush_after_run)
        self._ordered_rules = ordered_rules

    def compute(self) -> None:
        self.input._update_to_current()
        if self._array_inputs:
            self.cache = False
            self._clear_outputs()
        if self.cache is not False and self.unique_id in self._calculated:
            for consequent in self.ctrl.consequents:
                if _sim_state(consequent.output)[self] is not None:
                    self.output[consequent.label] = _sim_state(consequent.output)[self]
            return
        for antecedent in self.ctrl.antecedents:
            if _sim_state(antecedent.input)[self] is None:
                raise ValueError("All antecedents must have input values!")
            CrispValueCalculator(antecedent, self).fuzz(_sim_state(antecedent.input)[self])
        first = True
        for rule in self._ordered_rules:
            if first:
                for consequent in rule.consequent:
                    _sim_state(consequent.term.membership_value)[self] = None
                    _sim_state(consequent.activation)[self] = None
                first = False
            self.compute_rule(rule)
        self.output = self.defuzz_consequents()
        if self.cache is not False:
            self._calculated.append(self.unique_id)
        else:
            self._reset_simulation()
        self._run += 1
        if self._run % self._flush_after_run == 0:
            self._reset_simulation()


class FuzzySystem:
    def __init__(
        self, parameters: dict[str, dict[str, float]] | None = None
    ) -> None:
        self.parameters: dict[str, dict[str, float]] = (
            parameters if parameters is not None else DEFAULT_PARAMETERS
        )

        self.antecedents: dict[str, Antecedent] = {}

        for variable in FUZZY_VARIABLES:
            spec: VariableSpec = VARIABLE_SPECS[variable]
            universe: FloatArray = np.linspace(spec["min"], spec["max"], UNIVERSE_POINTS)
            antecedent = Antecedent(universe, variable)
            cut: dict[str, float] = self.parameters[variable]

            antecedent["low"] = fuzz.trapmf(
                universe, [spec["min"], spec["min"], cut["p1"], cut["p2"]]
            )
            antecedent["medium"] = fuzz.trimf(universe, [cut["p1"], cut["p2"], cut["p3"]])
            antecedent["high"] = fuzz.trapmf(
                universe, [cut["p2"], cut["p3"], spec["max"], spec["max"]]
            )

            self.antecedents[variable] = antecedent

        risk_universe: FloatArray = np.linspace(RISK_MIN, RISK_MAX, UNIVERSE_POINTS)
        self.consequent = Consequent(risk_universe, "risk")

        for term, points in OUTPUT_MEMBERSHIPS.items():
            if len(points) == 4:
                self.consequent[term] = fuzz.trapmf(risk_universe, points)
            else:
                self.consequent[term] = fuzz.trimf(risk_universe, points)

        self.rules: list[Rule] = []

        for entry in FUZZY_RULES:
            self.rules.append(self._build_rule(entry))

        if not self.rules:
            raise RuntimeError(MISSING_RULES_MESSAGE)

        self.system: ControlSystem = ControlSystem(self.rules)
        self._ordered_rules: list[Rule] = list(self.system.rules)

        self._rule_index: dict[str, int] = {}

        for index, rule in enumerate(self.rules):
            self._rule_index[str(rule.label)] = index

        self._rule_antecedents: dict[str, tuple[tuple[str, str], ...]] = {}

        for entry in FUZZY_RULES:
            self._rule_antecedents[str(entry["name"])] = tuple(
                cast(tuple[tuple[str, str], ...], entry["antecedents"])
            )

    def _build_rule(self, entry: dict[str, object]) -> Rule:
        antecedents: list[tuple[str, str]] = cast(
            list[tuple[str, str]], entry["antecedents"]
        )

        terms = []

        for variable, term in antecedents:
            terms.append(self.antecedents[variable][term])

        expression = terms[0]

        for term in terms[1:]:
            expression = expression & term

        return Rule(
            expression,
            self.consequent[cast(str, entry["consequent"])],
            label=str(entry["name"]),
        )

    def evaluate_batch(
        self, values: dict[str, FloatArray]
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        n: int = len(next(iter(values.values())))

        risk: FloatArray = np.zeros(n)
        rule_strengths: FloatArray = np.zeros((n, len(self.rules)))
        term_strengths: FloatArray = np.zeros((n, len(OUTPUT_TERMS)))
        term_index: dict[str, int] = {}

        for i, term in enumerate(OUTPUT_TERMS):
            term_index[term] = i

        simulation = FuzzySimulation(self.system, self._ordered_rules, n + 1)

        for i in range(n):
            for variable in FUZZY_VARIABLES:
                simulation.input[variable] = float(values[variable][i])
            simulation.compute()

            raw_risk: float = (
                float(simulation.output["risk"]) if "risk" in simulation.output else RISK_MIN
            )
            risk[i] = raw_risk if math.isfinite(raw_risk) else RISK_MIN

            for rule in self._ordered_rules:
                strength: float = _sim_float(rule.aggregate_firing, simulation)
                column: int = self._rule_index[str(rule.label)]
                rule_strengths[i, column] = strength
                output_term: str = rule.consequent[0].term.label
                term_strengths[i, term_index[output_term]] = max(
                    term_strengths[i, term_index[output_term]], strength
                )

        return risk, rule_strengths, term_strengths

    def predict(self, row: dict[str, float]) -> dict[str, object]:
        simulation = FuzzySimulation(self.system, self._ordered_rules, 2)

        for variable in FUZZY_VARIABLES:
            simulation.input[variable] = row[variable]
        simulation.compute()

        raw_risk: float = (
            float(simulation.output["risk"]) if "risk" in simulation.output else RISK_MIN
        )
        risk: float = raw_risk if math.isfinite(raw_risk) else RISK_MIN

        memberships: dict[str, dict[str, float]] = {}

        for variable in FUZZY_VARIABLES:
            memberships[variable] = {}

            for term in TERMS:
                membership: float = float(
                    self.antecedents[variable].terms[term].membership_value[simulation]
                )
                memberships[variable][term] = round(membership, 3)

        activated_rules: list[dict[str, object]] = []

        for rule in self._ordered_rules:
            strength: float = _sim_float(rule.aggregate_firing, simulation)

            if strength > 0.0:
                activated_rules.append(
                    {
                        "name": rule.label,
                        "antecedents": self._rule_antecedents[str(rule.label)],
                        "strength": round(strength, 3),
                    }
                )

        output_strengths: dict[str, float] = {}

        for term in OUTPUT_TERMS:
            output_strengths[term] = 0.0

        for rule in self._ordered_rules:
            strength_out: float = _sim_float(rule.aggregate_firing, simulation)
            output_term: str = rule.consequent[0].term.label
            output_strengths[output_term] = max(output_strengths[output_term], strength_out)

        output_rounded: dict[str, float] = {}

        for term, strength in output_strengths.items():
            output_rounded[term] = round(strength, 3)

        return {
            "risk_score": round(risk, 1),
            "memberships": memberships,
            "activated_rules": activated_rules,
            "output_strengths": output_rounded,
        }