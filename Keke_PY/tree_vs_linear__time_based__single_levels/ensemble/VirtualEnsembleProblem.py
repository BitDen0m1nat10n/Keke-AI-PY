import math
import multiprocessing
from copy import deepcopy
from typing import Optional, Dict, Tuple, Union, List

import numpy as np

from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.DummyRepresentation import DummyRepresentation
from Keke_PY.keke_game.keke import GameState, make_level, parse_map
from Keke_PY.search_agents.HeuristicGuidedSearch import HeuristicGuidedSearch
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.ClassifierEnsemble import ClassifierEnsemble
from Keke_PY.tree_vs_linear__time_based__single_levels.experiment_logs.read_in_combined_log import testing_levels, \
    times_by_genome_and_level



class VirtualEnsembleProblem(KekeProblem):

    times_for_policy_and_level: Dict[Tuple[str, str], Optional[float]]

    def __init__(
            self,
            times_for_policy_and_level: Dict[Tuple[str, str], Optional[float]] = None,
            **args
    ):
        if times_for_policy_and_level is None:
            times_for_policy_and_level = times_by_genome_and_level
        self.times_for_policy_and_level = deepcopy(times_for_policy_and_level)
        super().__init__(**args)
        assert self.max_node_expansions is None, \
            f"calculations have been done without restrictions on node-expansions, so limiting to {self.max_node_expansions} does yield false results"
        assert self.time_dependent_performance_function, \
            f"node-expansions of pre-calculations are not included in the combined data file, so having a performance-function depend on them does yield false results"
        assert self.max_calculation_time is not None and self.max_calculation_time <= 60.0, \
            f"calculation times longer than 60s have not been pre-calculated, so limiting to {self.max_calculation_time} does yield false results"
        if self.max_calculation_time != 60.0:
            for key, value in self.times_for_policy_and_level.items():
                if value is not None and value >= self.max_calculation_time:
                    self.times_for_policy_and_level[key] = None

    def register_and_run_next_generation(self, instances: [np.ndarray]):

        if self.is_continuation(instances):
            self.log_line("RESTART FROM PREVIOUS GENERATION DETECTED => SIMULATION WILL BE SKIPPED; RESULTS FROM PREVIOUS SIMULATION ARE RETURNED TO CALLER")
            self.try_continue_from_last_generation = False
            return

        self.log_generation_data(list(instances))
        self.past_instances_by_gen_and_index.update(((self.generation, index), deepcopy(instance)) for index, instance in enumerate(instances))

        self.fake_run_as_next_generation([
            ClassifierEnsemble.checked(self.representation.into_heuristic(instance))
            for instance in instances
        ])

    def fake_run_as_next_generation(self, instances: List[ClassifierEnsemble]):
        simulation_results: Dict[Tuple[int, str], Tuple[Union[List[str], None], int, float]] = {}
        for level_str in self.all_levels:
            initial_level_state: GameState = make_level(parse_map(level_str))
            for i, classifier_ensemble in enumerate(instances):
                selected_policy_genome: str = classifier_ensemble.get_heuristic_genome(initial_level_state)
                stored_solving_time: Optional[float] = self.times_for_policy_and_level[(selected_policy_genome, level_str)]
                fake_solution: Optional[List[str]] = None if stored_solving_time is None else ["NOT STORED"] # TODO: log real solution
                fake_node_expansions: int = math.nan # TODO: log real node_expansions
                simulation_results[(i, level_str)] = (
                    fake_solution,
                    fake_node_expansions,
                    self.max_calculation_time if stored_solving_time is None else stored_solving_time
                )

        self.past_evaluations_by_gen_index_and_level_id.update(
            ((self.generation, key[0], self.level_to_id_map[key[1]]), result)
            for key, result in simulation_results.items()
        )
        self.log_simulation_data(simulation_results)

        self.generation += 1









def evaluate_ensemble(
        ensemble: ClassifierEnsemble,
        time_limit: float = 60.0,
        logging_prefix: Optional[str] = "",
        virtual_evaluation_on_precomputed_data: bool = False
) -> str:
    representation = DummyRepresentation(ensemble)
    test_problem: KekeProblem
    if virtual_evaluation_on_precomputed_data:
        test_problem = VirtualEnsembleProblem.default_problem(
            representation=representation,
            executor=multiprocessing.Pool(20),
            max_node_expansions=None,
            max_calculation_time=time_limit,
            time_dependent_performance_function=True,
            agent_factory=HeuristicGuidedSearch.GuidedSearchFactory(),
            test_levels_or_src=testing_levels,
            training_levels_or_src=[],
            logging_prefix=logging_prefix
        )
    else:
        test_problem = KekeProblem.default_problem(
            representation=representation,
            executor=multiprocessing.Pool(20),
            max_node_expansions=None,
            max_calculation_time=time_limit,
            time_dependent_performance_function=True,
            agent_factory=HeuristicGuidedSearch.GuidedSearchFactory(),
            test_levels_or_src=testing_levels,
            training_levels_or_src=[],
            logging_prefix=logging_prefix,
        )
    return evaluate_problem(test_problem)


def evaluate_problem(
        test_problem: KekeProblem
) -> str:
    assert test_problem.representation.__class__ == DummyRepresentation
    assert test_problem.generation == 0
    test_problem.register_and_run_next_generation([test_problem.representation.deserialize("_")])
    performance_of_instance_on_batch: Dict[Tuple[int, int], float] = test_problem.get_performance_of_instance_on_batch()

    performances: Dict[int, float] = dict((key[1], value) for key, value in performance_of_instance_on_batch.items())

    if test_problem.logging_prefix is not None:
        print(test_problem.logging_prefix + f"performance: {performances}")

    data: Dict[Tuple[int, int, int], Tuple[Union[List[str], None], int, float]] = test_problem.past_evaluations_by_gen_index_and_level_id

    level_count: int = len(data.items())
    solved_level_count: int = sum(solution is not None for _, (solution, _, _) in data.items())
    total_node_expansions: int = sum(node_expansions for _, (_, node_expansions, _) in data.items())
    total_calculation_time: float = sum(calculation_time for _, (_, _, calculation_time) in data.items())

    if test_problem.logging_prefix is not None:
        print(test_problem.logging_prefix + f"level_count:{level_count}, solved_level_count:{solved_level_count}, total_node_expansions:{total_node_expansions}, total_calculation_time:{total_calculation_time}")
        print(test_problem.logging_prefix + f"solving ratio:{solved_level_count / level_count}, avg node expansions:{total_node_expansions / level_count}, avg time per level:{total_calculation_time / level_count}")

    return f"{solved_level_count}/{len(data.items())}"