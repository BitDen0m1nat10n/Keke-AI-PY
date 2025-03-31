import math
import multiprocessing
from typing import Tuple, Dict, Union, List

from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.DummyRepresentation import DummyRepresentation
from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.heuristics.SimpleHeuristic import SimpleHeuristic
from Keke_PY.heuristics.ZeroHeuristic import ZeroHeuristic
from Keke_PY.search_agents.HeuristicGuidedSearch import HeuristicGuidedSearch
from Keke_PY.search_agents.ai_interface import AgentFromPolicy


def eval_single_heuristic(heur: Heuristic, agent_factory: AgentFromPolicy = HeuristicGuidedSearch.GuidedSearchFactory()) -> Dict[int, float]:
    representation = DummyRepresentation(heur)
    test_problem = KekeProblem.default_problem(
        representation,
        multiprocessing.Pool(4),
        max_node_expansions=2000,
        max_calculation_time=math.inf,
        time_dependent_performance_function=False,
        agent_factory=agent_factory,
        #test_levels_or_src="./json_levels/full_biy_LEVELS.json",
        #training_levels_or_src=[],
    )
    # Level-2000-node-expansions-Time: "____________________\n_BWw..............._\n_11w.....G10......._\n_26w..wwwwwwww....._\n_www..woooooow....._\n_.....wooR15ow....._\n_.....woooooow....._\n_.....woborrow....._\n_.....woooooow....._\n_..wwwwwggggwwwww.._\n_..w............w.._\n_..w............w.._\n_..w............w.._\n_..wggg.........w.._\n_..wggg.....F13.w.._\n_..wfgg.........w.._\n_..wwwwwwwwwwwwww.._\n_.................._\n_.................._\n____________________"
    # TODO: check for possible time-improvement?
    # 56.54069662094116
    # 55.86849093437195
    # 56.63122844696045
    test_problem.register_and_run_next_generation([representation.deserialize("_")])
    performance_of_instance_on_batch: Dict[Tuple[int, int], float] = test_problem.get_performance_of_instance_on_batch()

    performances: Dict[int, float] = dict((key[1], value) for key, value in performance_of_instance_on_batch.items())

    print(performances)

    if True:
        data: Dict[Tuple[int, int, int], Tuple[Union[List[str], None], int]] = test_problem.past_evaluations_by_gen_index_and_level_id

        level_count: int = len(data.items())
        solved_level_count: int = sum(solution is not None for _, (solution, _, _) in data.items())
        total_node_expansions: int = sum(node_expansions for _, (_, node_expansions, _) in data.items())
        total_calculation_time: float = sum(calculation_time for _, (_, _, calculation_time) in data.items())
        print(level_count, solved_level_count, total_node_expansions, total_calculation_time)
        print(solved_level_count / level_count, total_node_expansions / level_count, total_calculation_time / level_count)


    return performances

if __name__ == "__main__":

    eval_single_heuristic(SimpleHeuristic())
