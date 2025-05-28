from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Union

import numpy as np

from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.HeuristicRepresentation import HeuristicRepresentation
from Keke_PY.heuristic_pymoo_representations.HeuristicTreeRepresentation import HeuristicTreeRepresentation
from Keke_PY.heuristic_pymoo_representations.TupleRepresentation import TupleRepresentation
from Keke_PY.heuristic_pymoo_representations.WeightedHeuristicSumRepresentation import \
    WeightedHeuristicSumRepresentation
from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.keke_game.keke import GameState, make_level, parse_map
from Keke_PY.search_agents.HeuristicGuidedSearch import HeuristicGuidedSearch
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.minimal_solving_set import get_minimal_solving_genomes
from Keke_PY.tree_vs_linear__time_based__single_levels.experiment_logs.read_in_combined_log import \
    times_by_genome_and_level, all_levels

# Train Ensemble
"""
def get_first_heuristic_value(serialized_heuristic: str, level_nr: int) -> float:
    representation: HeuristicRepresentation = HeuristicTreeRepresentation() if use_trees else WeightedHeuristicSumRepresentation()
    heuristic: Heuristic = representation.into_heuristic(representation.deserialize(serialized_heuristic))
    initial_state: GameState = make_level(parse_map(all_levels[level_nr]))
    ctx = {
        "initial GameState": initial_state,
        "initial rules": set(initial_state.rules)
    }
    return heuristic.run(initial_state, ctx)
"""

class EnsembleProblem(KekeProblem):

    ensemble_parts: int
    tuple_representation: TupleRepresentation
    inner_representation: HeuristicRepresentation
    level_and_times_for_policy: List[Tuple[str, Dict[str, Optional[float]]]]

    def __init__(
            self,
            representation: HeuristicRepresentation,
            level_and_times_for_policy: List[Tuple[str, Dict[str, Optional[float]]]],
    ):
        self.inner_representation = representation
        self.ensemble_parts = len(level_and_times_for_policy)
        self.tuple_representation = TupleRepresentation(representation, self.ensemble_parts)
        self.level_and_times_for_policy = level_and_times_for_policy
        super().__init__(
            training_batches = [list(level_and_times_for_policy[0][1].keys())],
            representation = self.tuple_representation,
            max_node_expansions = None,
            max_calculation_time = 60.0,
            time_dependent_performance_function = False,
            executor = ProcessPoolExecutor(),
            test_batch = [],
            agent_factory = HeuristicGuidedSearch.GuidedSearchFactory(),
            silent = False,
            logging_prefix = "",
        )

    def register_and_run_next_generation(self, instances: [np.ndarray]):

        if self.is_continuation(instances):
            self.log_line("RESTART FROM PREVIOUS GENERATION DETECTED => SIMULATION WILL BE SKIPPED; RESULTS FROM PREVIOUS SIMULATION ARE RETURNED TO CALLER")
            self.try_continue_from_last_generation = False
            return

        self.log_generation_data(list(instances))
        self.past_instances_by_gen_and_index.update(((self.generation, index), deepcopy(instance)) for index, instance in enumerate(instances))

        self.fake_run_as_next_generation([
            self.tuple_representation.into_heuristics(instance)
            for instance in instances
        ])

    def fake_run_as_next_generation(self, instances: List[List[Heuristic]]):
        simulation_results: Dict[Tuple[int, str], Tuple[Union[List[str], None], int, float]] = {}
        for level_str in self.all_levels:
            initial_level_state: GameState = make_level(parse_map(level_str))
            ctx = {
                "initial GameState": initial_level_state,
                "initial rules": set(initial_level_state.rules)
            }
            for i, ensemble_parts in enumerate(instances):
                priorities: List[float] = [
                    classifier_part.run(initial_level_state, ctx)
                    for classifier_part in ensemble_parts
                ]
                chosen_policy: int = priorities.index(max(priorities))
                stored_solving_time: Optional[float] = self.level_and_times_for_policy[chosen_policy][1][level_str]
                fake_solution: Optional[List[str]] = None if stored_solving_time is None else ["NOT STORED"] # TODO: log real solution
                fake_node_expansions: int = None # TODO: log real node_expansions
                simulation_results[(i, level_str)] = (fake_solution, fake_node_expansions, stored_solving_time)

        self.past_evaluations_by_gen_index_and_level_id.update(((self.generation, key[0], self.level_to_id_map[key[1]]), result) for key, result in simulation_results.items())
        self.log_simulation_data(simulation_results)

        self.generation += 1




use_trees: bool = True # TODO: make function argument!
only_use_training_levels: bool = True # TODO: make function argument!


minimal_solving_genomes: List[str] = get_minimal_solving_genomes(use_trees, only_use_training_levels, False)

level_and_times_for_policy: List[Tuple[str, Dict[str, Optional[float]]]] = [
    (genome, dict((level, times_by_genome_and_level[(genome, level)]) for level in all_levels))
    for genome in minimal_solving_genomes
]

representation: HeuristicRepresentation = HeuristicTreeRepresentation(3) if use_trees else WeightedHeuristicSumRepresentation()

ensemble_problem: EnsembleProblem = EnsembleProblem(representation, level_and_times_for_policy)

