if True:
    """Include Project root as Environment paths:"""
    from os.path import dirname, abspath
    import sys
    sys.path.append(dirname(dirname(dirname(dirname(abspath(__file__))))))

import multiprocessing

from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.DummyRepresentation import DummyRepresentation
from Keke_PY.heuristics.ZeroHeuristic import ZeroHeuristic
from Keke_PY.search_agents.BFS import BFS
from Keke_PY.search_agents.DFS import DFS
from Keke_PY.search_agents.ai_interface import AgentFromPolicy
from Keke_PY.tree_vs_linear__time_based__single_levels.experiment_logs.read_in_combined_log import testing_levels

import itertools
from typing import List, Union, Dict, Iterable, Optional
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.VirtualEnsembleProblem import evaluate_problem



def baseline_evaluation_run(
        baseline_methode: Union[
            "BFS",
            "DFS"
        ] = "BFS",
        max_calculation_time: float = 60.0,
        max_node_expansions: Optional[int] = None,
) -> str:

    baseline_factory: AgentFromPolicy
    if baseline_methode == "BFS":
        baseline_factory = BFS.BFSFactory()
    elif baseline_methode == "DFS":
        baseline_factory = DFS.DFSFactory()
    else:
        assert False, f'baseline_methode == \"{baseline_methode}\", but should be one of "BFS" or "DFS"!'

    test_problem = KekeProblem.default_problem(
        representation=DummyRepresentation(ZeroHeuristic()),
        executor=multiprocessing.Pool(20),
        max_node_expansions=max_node_expansions,
        max_calculation_time=max_calculation_time,
        time_dependent_performance_function=True,
        agent_factory=baseline_factory,
        test_levels_or_src=testing_levels,
        training_levels_or_src=[],
        logging_prefix="",
    )
    return evaluate_problem(test_problem)




if __name__ == "__main__":

    int_arguments: List[int] = []
    for argument in sys.argv:
        if argument.isdigit():
            int_arguments.append(int(argument))



    def get_kwargs(**kwargs):
        return kwargs
    argument_combos: List = [
        get_kwargs(
            baseline_methode = baseline_methode,
            max_calculation_time = max_calculation_time,
            max_node_expansions = max_node_expansions,
        )
        for \
            baseline_methode,\
            max_calculation_time, max_node_expansions
            in itertools.product(
                ["BFS", "DFS"],
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
        res.append(baseline_evaluation_run(**argument_combo))
        results[combo_index] = res

    for combo_index in combo_indices:
        print(argument_combos[combo_index])
        print(results[combo_index])

