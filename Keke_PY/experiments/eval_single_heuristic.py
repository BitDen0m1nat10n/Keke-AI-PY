import multiprocessing
from typing import Tuple, Dict

from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.DummyRepresentation import DummyRepresentation
from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.heuristics.ZeroHeuristic import ZeroHeuristic
from Keke_PY.search_agents.HeuristicGuidedSearch import HeuristicGuidedSearch
from Keke_PY.search_agents.ai_interface import AgentFromPolicy


def eval_single_heuristic(heur: Heuristic, agent_factory: AgentFromPolicy = HeuristicGuidedSearch.GuidedSearchFactory()) -> Dict[int, float]:
    representation = DummyRepresentation(heur)
    test_problem = KekeProblem.default_problem(representation, multiprocessing.Pool(), agent_factory=agent_factory)

    test_problem.register_and_run_next_generation([representation.deserialize("_")])
    performance_of_instance_on_batch: Dict[Tuple[int, int], float] = test_problem.get_performance_of_instance_on_batch()

    performances: Dict[int, float] = dict((key[1], value) for key, value in performance_of_instance_on_batch.items())

    print(performances)
    return performances

if __name__ == "__main__":

    eval_single_heuristic(ZeroHeuristic())
