from abc import ABC, abstractmethod
from typing import final

from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.keke_game.keke import GameState


class ClassifierEnsemble(Heuristic, ABC):

    @abstractmethod
    def get_heuristic(self, initial_game_state: GameState) -> Heuristic:
        pass

    @final
    def run(self, state: GameState, ctx: dict, *args: float) -> float:
        selected_ensemble_part: Heuristic = ctx["ensemble_part"]
        ctx_for_ensemble_part: dict = ctx["ctx"]
        return selected_ensemble_part.run(state, ctx_for_ensemble_part)

    @final
    def get_ctx(self, initial_game_state: GameState) -> dict:
        selected_ensemble_part: Heuristic = self.get_heuristic(initial_game_state)
        ctx_for_ensemble_part: dict = selected_ensemble_part.get_ctx(initial_game_state)
        return {
            "ensemble_part": selected_ensemble_part,
            "ctx": ctx_for_ensemble_part
        }