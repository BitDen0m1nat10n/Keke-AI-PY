from abc import ABC, abstractmethod
from typing import final, Dict, List

from Keke_PY.heuristic_pymoo_representations.HeuristicRepresentation import HeuristicRepresentation
from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.keke_game.keke import GameState


class ClassifierEnsemble(Heuristic, ABC):

    _precomputed_heuristic: Dict[str, Heuristic]

    def __init__(
            self,
            precomputed_heuristics: Dict[str, Heuristic],
    ):
        self._precomputed_heuristic = precomputed_heuristics

    @abstractmethod
    def get_heuristic_genome(self, initial_game_state: GameState) -> str:
        pass

    def get_heuristic(self, initial_game_state: GameState) -> Heuristic:
        return self._precomputed_heuristic[
            self.get_heuristic_genome(initial_game_state)
        ]

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

    @staticmethod
    def precompute_heuristics(
            genomes: List[str],
            representation: HeuristicRepresentation
    ) -> Dict[str, Heuristic]:
        return dict([
            (genome, representation.into_heuristic(
                representation.deserialize(genome)
            ))
            for genome in genomes
        ])

    @staticmethod
    def checked(
            heuristic: Heuristic
    ) -> 'ClassifierEnsemble':
        assert isinstance(heuristic, ClassifierEnsemble), \
            f"{heuristic} should be of type 'ClassifierEnsemble', but is {heuristic.__class__}!"
        return heuristic