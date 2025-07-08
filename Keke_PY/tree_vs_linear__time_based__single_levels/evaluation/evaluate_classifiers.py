if True:
    """Include Project root as Environment paths:"""
    from os.path import dirname, abspath
    import sys
    sys.path.append(dirname(dirname(dirname(dirname(abspath(__file__))))))



import itertools
from typing import List, Tuple, Union, Dict, Iterable, Optional

from sklearn.ensemble import RandomForestClassifier

from Keke_PY.heuristic_pymoo_representations.HeuristicRepresentation import HeuristicRepresentation
from Keke_PY.heuristic_pymoo_representations.HeuristicTreeRepresentation import HeuristicTreeRepresentation
from Keke_PY.heuristic_pymoo_representations.WeightedHeuristicSumRepresentation import \
    WeightedHeuristicSumRepresentation
from Keke_PY.keke_game.keke import GameState, make_level, parse_map
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.ClassifierEnsemble import ClassifierEnsemble
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.VirtualEnsembleProblem import evaluate_ensemble
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.minimal_solving_set import get_minimal_solving_genomes
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.NearestNeighborEnsemble import NearestNeighborEnsemble
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.HeuristicClassifierEnsemble import HeuristicClassifierEnsemble
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.RandomForestClassifierEnsemble import \
    RandomForestClassifierEnsemble, get_random_forest_classifier
from Keke_PY.tree_vs_linear__time_based__single_levels.experiment_logs.read_in_combined_log import get_genomes, \
    all_levels


def evaluation_run(
        use_trees: bool = True,
        limit_to_minimal_solving_set: bool = True,
        virtual_evaluation_on_precomputed_data: bool = False,
        ensemble_method: Union[
            "NearestNeighborEnsemble",
            "RandomForestClassifierEnsemble",
            "HeuristicClassifierEnsemble"
        ] = "RandomForestClassifierEnsemble",
        max_calculation_time: float = 60.0,
        max_node_expansions: Optional[int] = None,
) -> str:

    policy_representation: HeuristicRepresentation = HeuristicTreeRepresentation() if use_trees else WeightedHeuristicSumRepresentation()

    available_genomes: List[str] = get_genomes(use_trees, True, False)
    if limit_to_minimal_solving_set:
        available_genomes = get_minimal_solving_genomes(use_trees, True, True)


    ensemble: ClassifierEnsemble
    if ensemble_method == "NearestNeighborEnsemble":
        heuristic_genomes_for_level: List[Tuple[GameState, str]] = [
            (make_level(parse_map(level_str)), genome)
            for level_str, genome in zip(all_levels, get_genomes(use_trees, True, True))
        ]
        ensemble = NearestNeighborEnsemble(
            heuristic_genomes_for_level=heuristic_genomes_for_level,
            representation=policy_representation
        )
    elif ensemble_method == "RandomForestClassifierEnsemble":
        classifier: RandomForestClassifier = get_random_forest_classifier(
            available_genomes,
            False
        )
        ensemble = RandomForestClassifierEnsemble(
            classifier,
            policy_representation
        )
    elif ensemble_method == "HeuristicClassifierEnsemble":



        assert False, "TODO: virtual training of classifiers"
    else:
        assert False, f'ensemble_method == \"{ensemble_method}\", but should be one of ' + \
                      '"NearestNeighborEnsemble", "RandomForestClassifierEnsemble" or "HeuristicClassifierEnsemble"'

    return evaluate_ensemble(
        ensemble,
        virtual_evaluation_on_precomputed_data=virtual_evaluation_on_precomputed_data,
        max_calculation_time=max_calculation_time,
        max_node_expansions=max_node_expansions,
    )




if __name__ == "__main__":

    int_arguments: List[int] = []
    for argument in sys.argv:
        if argument.isdigit():
            int_arguments.append(int(argument))



    def get_kwargs(**kwargs):
        return kwargs
    argument_combos: List = [
        get_kwargs(
            use_trees = use_trees,
            limit_to_minimal_solving_set = limit_to_minimal_solving_set,
            virtual_evaluation_on_precomputed_data = int_arguments[0],
            ensemble_method = ensemble_method,
            max_calculation_time = max_calculation_time,
            max_node_expansions = max_node_expansions,
        )
        for use_trees, limit_to_minimal_solving_set,\
            ensemble_method,\
            max_calculation_time, max_node_expansions
            in itertools.product(
                [True, False],
                [True, False],
                ["NearestNeighborEnsemble", "RandomForestClassifierEnsemble"],#, "HeuristicClassifierEnsemble"]
                [60.0, 10.0],
                [None, 10_000]
            )
        if (max_calculation_time == 60.0 and max_node_expansions is None) or\
           (max_calculation_time == 10.0 and max_node_expansions is 10_000)
    ]

    results: Dict[int, List[str]] = {}

    combo_indices: Iterable[int] = range(len(argument_combos))

    for combo_index in combo_indices:
        argument_combo = argument_combos[combo_index]
        print(argument_combo)
        res: List[str] = []

        print(argument_combo)
        res.append(evaluation_run(**argument_combo))
        results[combo_index] = res

    for combo_index in combo_indices:
        print(argument_combos[combo_index])
        print(results[combo_index])

