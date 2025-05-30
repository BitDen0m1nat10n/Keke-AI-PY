if True:
    """Include Project root as Environment paths:"""
    from os.path import dirname, abspath
    import sys
    sys.path.append(dirname(dirname(dirname(abspath(__file__)))))


import math
import multiprocessing
from typing import Tuple, Dict, Union, List, Optional

from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.DummyRepresentation import DummyRepresentation
from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.heuristics.SimpleHeuristic import SimpleHeuristic
from Keke_PY.search_agents.HeuristicGuidedSearch import HeuristicGuidedSearch
from Keke_PY.search_agents.ai_interface import AgentFromPolicy


def eval_single_heuristic(
        heur: Heuristic,
        test_levels_or_src: Union[List[str], str] = "./json_levels/full_biy_LEVELS.json",
        agent_factory: AgentFromPolicy = HeuristicGuidedSearch.GuidedSearchFactory(),
        logging_prefix: Optional[str] = "",
) -> Dict[int, float]:
    representation = DummyRepresentation(heur)
    test_problem = KekeProblem.default_problem(
        representation,
        multiprocessing.Pool(20),
        max_node_expansions=None,
        max_calculation_time=60.0,
        time_dependent_performance_function=True,
        agent_factory=agent_factory,
        test_levels_or_src=test_levels_or_src,
        training_levels_or_src=[],
        logging_prefix=logging_prefix,
    )
    # Level-2000-node-expansions-Time: "____________________\n_BWw..............._\n_11w.....G10......._\n_26w..wwwwwwww....._\n_www..woooooow....._\n_.....wooR15ow....._\n_.....woooooow....._\n_.....woborrow....._\n_.....woooooow....._\n_..wwwwwggggwwwww.._\n_..w............w.._\n_..w............w.._\n_..w............w.._\n_..wggg.........w.._\n_..wggg.....F13.w.._\n_..wfgg.........w.._\n_..wwwwwwwwwwwwww.._\n_.................._\n_.................._\n____________________"
    # TODO: check for possible time-improvement?
    # 56.54069662094116
    # 55.86849093437195
    # 56.63122844696045
    test_problem.register_and_run_next_generation([representation.deserialize("_")])
    performance_of_instance_on_batch: Dict[Tuple[int, int], float] = test_problem.get_performance_of_instance_on_batch()

    performances: Dict[int, float] = dict((key[1], value) for key, value in performance_of_instance_on_batch.items())

    if logging_prefix is not None:
        print(logging_prefix + f"performance: {performances}")

        data: Dict[Tuple[int, int, int], Tuple[Union[List[str], None], int, float]] = test_problem.past_evaluations_by_gen_index_and_level_id

        level_count: int = len(data.items())
        solved_level_count: int = sum(solution is not None for _, (solution, _, _) in data.items())
        total_node_expansions: int = sum(node_expansions for _, (_, node_expansions, _) in data.items())
        total_calculation_time: float = sum(calculation_time for _, (_, _, calculation_time) in data.items())
        print(logging_prefix + f"level_count:{level_count}, solved_level_count:{solved_level_count}, total_node_expansions:{total_node_expansions}, total_calculation_time:{total_calculation_time}")
        print(logging_prefix + f"solving ratio:{solved_level_count / level_count}, avg node expansions:{total_node_expansions / level_count}, avg time per level:{total_calculation_time / level_count}")


    return performances

if __name__ == "__main__":

    eval_single_heuristic(SimpleHeuristic())

    """
    Test run on my pc
    
    performance: {-1: 75.637660521528, 0: nan}
    level_count:184, solved_level_count:121, total_node_expansions:191226, total_calculation_time:4382.670464038849
    solving ratio:0.657608695652174, avg node expansions:1039.2717391304348, avg time per level:23.81886121760244
    """