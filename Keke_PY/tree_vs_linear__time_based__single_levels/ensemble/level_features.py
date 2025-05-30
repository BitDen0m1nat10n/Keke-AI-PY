from typing import List, Tuple, Dict

import numpy as np

from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.heuristics.hand_crafted_heuristics import named_heuristics
from Keke_PY.keke_game.keke import GameState

level_feature_names_and_params: List[Tuple[str, Tuple[float, ...]]] = [
    ("number_of_goal_objects", ()),
    ("number_of_player_objects", ()),
    ("connectivity", ()),
    ("number_of_auto_movers", ()),

    ("number_of_stuck_is_words", ()),
    ("number_of_stuck_prefixes", ()),
    ("number_of_stuck_suffixes", ()),
    ("number_of_stuck_important_suffixes", ()),

    ("number_of_killer_objects", ()),
    ("number_of_pushable_objects", ()),
    ("number_of_sinkable_objects", ()),
    ("number_of_stopped_objects", ()),
    ("player_killer_distance", (-1.0,)),

    ("distance_to_winnable_objects", (-1.0,)),
    ("distance_to_words", (-1.0,)),
    ("distance_to_pushable_objects", (-1.0,)),

    ("number_of_newly_created_rules", ()),

    ("goal_reachability", ())
]

named_level_feature_heuristics: Dict[str, Heuristic] = dict([
    (name, dict(named_heuristics)[name].with_parameters(*parameters))
    for name, parameters in level_feature_names_and_params
])

def get_feature_value(initial_level_state: GameState, feature_heuristic: Heuristic) -> float:
    return feature_heuristic.run(initial_level_state, feature_heuristic.get_ctx(initial_level_state))

def get_feature_value_vector(initial_level_state: GameState) -> np.ndarray:
    return np.array([
        get_feature_value(initial_level_state, named_level_feature_heuristics[name])
        for name, _ in level_feature_names_and_params
    ]).reshape(1, -1)
