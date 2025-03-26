import multiprocessing
from typing import Tuple, Dict, Union, List

from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.DummyRepresentation import DummyRepresentation
from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.heuristics.ZeroHeuristic import ZeroHeuristic
from Keke_PY.search_agents.HeuristicGuidedSearch import HeuristicGuidedSearch
from Keke_PY.search_agents.ai_interface import AgentFromPolicy


def eval_single_heuristic(heur: Heuristic, agent_factory: AgentFromPolicy = HeuristicGuidedSearch.GuidedSearchFactory()) -> Dict[int, float]:
    representation = DummyRepresentation(heur)
    test_problem = KekeProblem.default_problem(representation, multiprocessing.Pool(), agent_factory=agent_factory, test_levels_or_src="./json_levels/full_biy_LEVELS.json", training_levels_or_src=[])

    test_problem.register_and_run_next_generation([representation.deserialize("_")])
    performance_of_instance_on_batch: Dict[Tuple[int, int], float] = test_problem.get_performance_of_instance_on_batch()

    performances: Dict[int, float] = dict((key[1], value) for key, value in performance_of_instance_on_batch.items())

    print(performances)

    if True:
        data: Dict[Tuple[int, int, int], Tuple[Union[List[str], None], int]] = test_problem.past_evaluations_by_gen_index_and_level_id

        level_count: int = len(data.items())
        solved_level_count: int = sum(solution is not None for _, (solution, _) in data.items())
        total_node_expansions: int = sum(node_expansions for _, (_, node_expansions) in data.items())
        print(level_count, solved_level_count, total_node_expansions)
        print(solved_level_count / level_count, total_node_expansions / level_count)


    return performances

if __name__ == "__main__":

    eval_single_heuristic(ZeroHeuristic())
