if True:
    """Include Project root as Environment paths:"""
    from os.path import dirname, abspath
    import sys
    sys.path.append(dirname(dirname(dirname(dirname(abspath(__file__))))))


import multiprocessing
import time
from typing import Iterable, List, Tuple

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.algorithm import Algorithm
from pymoo.optimize import minimize
import numpy as np

from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.VirtualEnsembleProblem import evaluate_ensemble

from Keke_PY.search_agents.HeuristicGuidedSearch import HeuristicGuidedSearch
from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.HeuristicTreeRepresentation import HeuristicTreeRepresentation
from Keke_PY.heuristic_pymoo_representations.TrackedRepresentation import TrackedRepresentation
from Keke_PY.heuristic_pymoo_representations.WeightedHeuristicSumRepresentation import WeightedHeuristicSumRepresentation
from Keke_PY.experiments.eval_single_heuristic import eval_single_heuristic
from Keke_PY.heuristics.SimpleHeuristic import SimpleHeuristic

int_arguments: List[int] = []
for argument in sys.argv:
    if argument.isdigit():
        int_arguments.append(int(argument))

pop_size: int = 10
n_generations: int = 50

n_evals: int = pop_size * n_generations


representation = TrackedRepresentation(
    HeuristicTreeRepresentation(5) if int_arguments[0] == 1 else WeightedHeuristicSumRepresentation(0.5)
)

agent_factory = HeuristicGuidedSearch.GuidedSearchFactory()

optimization_algorithm: Algorithm = GA(pop_size=pop_size, **representation.algorithm_arguments())

training_problem = KekeProblem.default_problem(
    representation,
    multiprocessing.Pool(20),
    agent_factory,
    training_levels_or_src = "./json_levels/train_LEVELS.json",
    test_levels_or_src = "./json_levels/test_LEVELS.json",
    limit_levels = None,
    max_calculation_time = 60.0,
    max_node_expansions = None,
    time_dependent_performance_function = True,
    logging_prefix="ACTUAL_TRAINING__"
)



def measure_time() -> Iterable[None]:
    start = time.time()
    yield None
    end = time.time()
    print("The time of execution is:", (end - start), "s")

if __name__ == '__main__':

    evaluate_ensemble(SimpleHeuristic(), 60.0, "BASELINE__", False)

    for _ in measure_time():
        info: List = [optimization_algorithm, representation, agent_factory, "60.0s/level"]

        print(info)

        for _ in measure_time():
            print(f"TRAINING ON all {len(training_problem.training_batches[0])} training_levels:")


            res = minimize(
                training_problem,
                optimization_algorithm,
                termination=("n_eval", n_evals),
                verbose=True
            )

            print(f"TRAINING ON all {len(training_problem.training_batches[0])} training_levels.")
            print("training done.")

        print("\nTRAINING DONE\n")
        print("\n-----!!!NEW PROBLEM!!!-----\n")
        print("\n---TEST BEST INDUVIDUAL ON ALL LEVELS---\n")

        best_individual_index: Tuple[int, int] = max(
            training_problem.get_best_past_individuals_generation_nrs_and_indices(0))
        best_individual_encoded: np.ndarray = training_problem.past_instances_by_gen_and_index[best_individual_index]
        best_individual: Heuristic = representation.into_heuristic(best_individual_encoded)

        evaluate_ensemble(best_individual, 60.0, "BASELINE__", False)
        print(info)
