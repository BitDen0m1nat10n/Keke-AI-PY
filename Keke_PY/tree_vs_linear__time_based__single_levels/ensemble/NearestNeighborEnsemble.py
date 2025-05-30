from typing import Tuple, List, Callable

import numpy as np

from Keke_PY.heuristic_pymoo_representations.HeuristicRepresentation import HeuristicRepresentation
from Keke_PY.keke_game.keke import GameState
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.ClassifierEnsemble import ClassifierEnsemble
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.level_features import get_feature_value_vector


class NearestNeighborEnsemble(ClassifierEnsemble):

    def get_heuristic_genome(self, initial_game_state: GameState) -> str:
        feature_vect: np.ndarray = self._get_feature_vector(initial_game_state)
        distances: List[float] = [
            float(np.linalg.norm(feature_vect - other_feature_vect))
            for other_feature_vect, _ in self._genomes_for_feature_vect
        ]
        return self._genomes_for_feature_vect[distances.index(min(distances))][1]

    _get_feature_vector: Callable[[GameState], np.ndarray]
    _genomes_for_feature_vect: List[Tuple[np.ndarray, str]]

    def __init__(
            self,
            heuristic_genomes_for_level: List[Tuple[GameState, str]],
            representation: HeuristicRepresentation,
            get_feature_vector: Callable[[GameState], np.ndarray] = get_feature_value_vector
    ):
        super().__init__(ClassifierEnsemble.precompute_heuristics([
            genome for _, genome in heuristic_genomes_for_level
        ], representation))
        self._genomes_for_feature_vect = [
            (get_feature_vector(level), heuristic)
            for level, heuristic in heuristic_genomes_for_level
        ]
        self._get_feature_vector = get_feature_vector