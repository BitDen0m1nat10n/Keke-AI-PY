from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Callable
from inspect import signature
from typing import final, Tuple

from Keke_PY.keke_game.keke import GameState

class ParametrisedHeuristic(ABC):

    @property
    @abstractmethod
    def nr_of_parameters(self) -> int:
        pass

    @abstractmethod
    def run(self, state: GameState, ctx: dict, *args: float) -> float:
        pass

    @staticmethod
    def from_func(heuristic: Callable, override_nr_of_parameters: int = -1) -> 'ParametrisedHeuristicFromCallable':
        return ParametrisedHeuristicFromCallable(heuristic, override_nr_of_parameters)

    def with_parameters(self, *parameters: float) -> 'ParametrisedHeuristicWithParameters':
        return ParametrisedHeuristicWithParameters(self, *parameters)


class Heuristic(ParametrisedHeuristic, ABC):

    @final
    @property
    def nr_of_parameters(self) -> int:
        return 0

    def get_ctx(self, initial_game_state: GameState) -> dict:
        return {
            "initial GameState": initial_game_state,
            "initial rules": set(initial_game_state.rules)
        }

@dataclass
class ParametrisedHeuristicFromCallable(ParametrisedHeuristic):
    heuristic_callable: Callable
    _nr_of_parameters: int

    def __init__(self, heuristic: Callable, override_nr_of_parameters: int = -1):
        self.heuristic_callable = heuristic
        if override_nr_of_parameters == -1:
            self._nr_of_parameters = len(signature(heuristic).parameters) - 2
        else:
            self._nr_of_parameters = override_nr_of_parameters

    @property
    def nr_of_parameters(self) -> int:
        return self._nr_of_parameters

    def run(self, state: GameState, ctx: dict, *args: float) -> float:
        return float(self.heuristic_callable(state, ctx, *args))

@dataclass
class ParametrisedHeuristicWithParameters(Heuristic):
    parametrised_heuristic: ParametrisedHeuristic
    parameters: Tuple[float, ...]

    def __init__(self, parametrised_heuristic: ParametrisedHeuristic, *parameters: float):
        assert parametrised_heuristic.nr_of_parameters == len(parameters), \
            f"Heuristic needs {parametrised_heuristic.nr_of_parameters} parameters. {len(parameters)} were given."
        self.parametrised_heuristic = parametrised_heuristic
        self.parameters = parameters

    def run(self, state: GameState, ctx: dict, *args: float) -> float:
        assert len(args) == 0, f"No parameters are expected. {args} are not supposed to be here."
        return self.parametrised_heuristic.run(state, ctx, *self.parameters)