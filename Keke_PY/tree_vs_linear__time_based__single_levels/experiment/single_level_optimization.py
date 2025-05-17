if True:
    """Include Project root as Environment paths:"""
    from os.path import dirname, abspath
    import sys
    sys.path.append(dirname(dirname(dirname(dirname(abspath(__file__))))))

import multiprocessing
import time
import numpy as np
from typing import Iterable, List, Tuple, Dict

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.algorithm import Algorithm
from pymoo.optimize import minimize

from Keke_PY.experiments.eval_single_heuristic import eval_single_heuristic
from Keke_PY.heuristic_pymoo_representations.WeightedHeuristicSumRepresentation import \
    WeightedHeuristicSumRepresentation
from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.heuristics.SimpleHeuristic import SimpleHeuristic
from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.HeuristicTreeRepresentation import HeuristicTreeRepresentation
from Keke_PY.heuristic_pymoo_representations.TrackedRepresentation import TrackedRepresentation
from Keke_PY.keke_game.simulation import load_level_set


int_arguments: List[int] = []
for argument in sys.argv:
    if argument.isdigit():
        int_arguments.append(int(argument))

levels: List[str] = [
    *[level["ascii"] for level in
      load_level_set("./json_levels/train_LEVELS.json")["levels"]],
    *[level["ascii"] for level in
      load_level_set("./json_levels/test_LEVELS.json")["levels"]],
]

level_nr: int = (int_arguments[0] // 2)
level: str = levels[level_nr]
use_trees: bool = (int_arguments[0] % 2) == 1

pop_size: int = 10
n_generations: int = 50


n_evals: int = pop_size * n_generations


representation = TrackedRepresentation(
    HeuristicTreeRepresentation(10) if use_trees else WeightedHeuristicSumRepresentation(0.5)
)

executor=multiprocessing.Pool(10)

optimization_algorithm: Algorithm = GA(pop_size=pop_size, **representation.algorithm_arguments())



def measure_time() -> Iterable[None]:
    start = time.time()
    print("Start measuring time.")
    yield None
    end = time.time()
    print("The time of execution is:", (end - start), "s")


run_info: Dict[str, object] = {
    "script": __file__,
    "algorithm": optimization_algorithm,
    "use_trees": use_trees,
    "representation": representation,
    "job_arr_index": (int_arguments[0] // 2),
    "level": level,
    "pop_size": pop_size,
    "n_gen": n_generations
}
info: str = ",\n".join(f"{name}: {obj}" for name, obj in run_info.items())

if __name__ == '__main__':

    print(info)

    eval_single_heuristic(SimpleHeuristic(), test_levels_or_src=[level], logging_prefix="BASELINE_ON_TRAINING_LEVEL__")

    for _ in measure_time():

        print(f"TRAINING ON LEVEL:{level_nr}")
        print(f"of {len(levels)} levels")
        print(f"---LEVEL STRING---\n{level}\n---LEVEL STRING---")

        training_problem: KekeProblem = KekeProblem(
            training_batches=[[level]],
            representation=representation,
            max_node_expansions=None,
            max_calculation_time=60.0,
            time_dependent_performance_function=True,
            executor=executor,
            test_batch=[],
            logging_prefix="TRAIN__"
        )

        res = minimize(
            training_problem,
            optimization_algorithm,
            termination=("n_eval", n_evals),
            verbose=True
        )


        print(f"TRAINING ON LEVEL:{level_nr}")
        print(f"of {len(levels)} levels")
        print(f"---LEVEL STRING---\n{level}\n---LEVEL STRING---")
        print("training done.")

    print("\nTRAINING DONE\n")
    print("\n-----!!!NEW PROBLEM!!!-----\n")
    print("\n---TEST BEST INDUVIDUAL ON ALL LEVELS---\n")

    best_individual_index: Tuple[int, int] = max(training_problem.get_best_past_individuals_generation_nrs_and_indices(0))
    best_individual_encoded: np.ndarray = training_problem.past_instances_by_gen_and_index[best_individual_index]
    best_individual: Heuristic = representation.into_heuristic(best_individual_encoded)

    test_levels: List[str] = [
        *[level["ascii"] for level in
          load_level_set("./json_levels/train_LEVELS.json")["levels"]],
        *[level["ascii"] for level in
          load_level_set("./json_levels/test_LEVELS.json")["levels"]],
    ]

    eval_single_heuristic(best_individual, test_levels_or_src=test_levels, logging_prefix="RESULT_ON_ALL_LEVELS__")

    print(info)
