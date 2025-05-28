import math
from random import shuffle
from typing import List, Optional, Tuple, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from Keke_PY.experiments.eval_single_heuristic import eval_single_heuristic
from Keke_PY.heuristic_pymoo_representations.HeuristicRepresentation import HeuristicRepresentation
from Keke_PY.heuristic_pymoo_representations.HeuristicTreeRepresentation import HeuristicTreeRepresentation
from Keke_PY.heuristic_pymoo_representations.WeightedHeuristicSumRepresentation import \
    WeightedHeuristicSumRepresentation
from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.heuristics.hand_crafted_heuristics import named_heuristics
from Keke_PY.keke_game.keke import parse_map, make_level, GameState
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.ClassifierEnsemble import ClassifierEnsemble
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.minimal_solving_set import get_minimal_solving_genomes
from Keke_PY.tree_vs_linear__time_based__single_levels.experiment_logs.read_in_combined_log import all_levels, \
    times_by_genome_and_level, get_levels


# build level features and ideal classifications

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


def get_best_genome_of_set_on_level_str(genome_str_set: List[str], level_str: str) -> str:
    genome_str_set = list(genome_str_set)  # to make sure, this is actually a list and not a set o.e.
    times: List[Optional[float]] = [
        times_by_genome_and_level[(genome_str, level_str)]
        for genome_str in genome_str_set
    ]
    penalty: List[float] = [math.inf if time is None else time for time in times]
    return genome_str_set[penalty.index(min(*penalty))]

def get_scikit_learn_data_for_level_strs(
        level_strs: List[str],
        best_genome_of_minimal_solving_set_label_for_level_str: Dict[str, str],
        shffl: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    level_strs = list(level_strs) # to make sure, this is actually a list and not a set o.e.
    # {array-like, sparse matrix} of shape (n_samples, n_features)
    feature_by_level_and_name: Dict[Tuple[str, str], float] = dict([
        ((level_str, feature_name), get_feature_value(make_level(parse_map(level_str)), feature_heuristic))
        for level_str in all_levels for feature_name, feature_heuristic in named_level_feature_heuristics.items()
    ])
    x_values: np.ndarray = np.array([
        [
            feature_by_level_and_name[(level_str, feature_name)]
            for feature_name, _ in level_feature_names_and_params
        ]
        for level_str in level_strs
    ])
    y_values: List[str] = [
        best_genome_of_minimal_solving_set_label_for_level_str[level_str]
        for level_str in level_strs
    ]
    if shffl:
        shuffle(y_values)
    return x_values, np.array(y_values)


def get_random_forest_classifier(
        genome_set: Optional[List[str]],
        shuffle_labels_for_baseline: bool = False,
        training_level_set: List[str] = 0,
        virtual_testing_level_set: Optional[List[str]] = 0,
) -> RandomForestClassifier:
    if genome_set is None:
        genome_set: List[str] = get_minimal_solving_genomes(True, True, False)
    if training_level_set.__class__ == int:
        training_level_set = get_levels(True, False)
    if virtual_testing_level_set.__class__ == int:
        virtual_testing_level_set = get_levels(False, True)


    best_genome_of_minimal_solving_set_label_for_level_str: Dict[str, str] = dict([
        (level_str, get_best_genome_of_set_on_level_str(genome_set, level_str))
        for level_str in all_levels
    ])


    train_x, train_y = get_scikit_learn_data_for_level_strs(
        training_level_set,
        best_genome_of_minimal_solving_set_label_for_level_str,
        shuffle_labels_for_baseline
    )

    classifier: RandomForestClassifier = RandomForestClassifier()

    classifier.fit(train_x, train_y)

    if virtual_testing_level_set is not None:
        test_x, y_test = get_scikit_learn_data_for_level_strs(
            virtual_testing_level_set,
            best_genome_of_minimal_solving_set_label_for_level_str,
            False,
        )

        y_pred = classifier.predict(test_x)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        solvable_test_level_count: int = len([
            test_level for test_level in virtual_testing_level_set
            if any(
                times_by_genome_and_level[(genome_str, test_level)] is not None
                for genome_str in genome_set
            )
        ])
        solved_test_level_count: int = len([
            test_level for i, test_level in enumerate(virtual_testing_level_set)
            if times_by_genome_and_level[(y_pred[i], test_level)] is not None
        ])
        print(f"{solved_test_level_count}/{solvable_test_level_count}/{len(virtual_testing_level_set)}")
        #print(y_pred)

    return classifier



class RandomForestClassifierEnsemble(ClassifierEnsemble):

    classifier: RandomForestClassifier
    ensemble_parts_by_classifier_label: Dict[str, Heuristic]

    def __init__(
            self,
            classifier: RandomForestClassifier,
            ensemble_parts_by_classifier_label: Dict[str, Heuristic],
    ):
        self.classifier = classifier
        self.ensemble_parts_by_classifier_label = ensemble_parts_by_classifier_label

    def get_heuristic(self, initial_game_state: GameState) -> Heuristic:
        label: str = self.classifier.predict(get_feature_value_vector(initial_game_state))[0]
        #print(label)
        return self.ensemble_parts_by_classifier_label[label]


if __name__ == "__main__":

    use_trees: bool = True
    representation: HeuristicRepresentation = HeuristicTreeRepresentation() if use_trees else WeightedHeuristicSumRepresentation()


    minimal_solving_genomes: List[str] = get_minimal_solving_genomes(use_trees, True, True)
    classifier: RandomForestClassifier = get_random_forest_classifier(
        minimal_solving_genomes,
        False
    )

    ensemble: RandomForestClassifierEnsemble = RandomForestClassifierEnsemble(
        classifier,
        dict(
            (genome, representation.into_heuristic(representation.deserialize(genome)))
            for genome in minimal_solving_genomes
        )
    )

    eval_single_heuristic(ensemble)

    """
    Test run on my pc
    
    performance: {-1: 54.755799522866376, 0: nan}
    level_count:184, solved_level_count:89, total_node_expansions:265879, total_calculation_time:6304.932887792587
    solving ratio:0.483695652173913, avg node expansions:1444.9945652173913, avg time per level:34.265939607568406
    """