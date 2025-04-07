from typing import Tuple, List, Dict, Callable

import numpy as np
from matplotlib import pyplot as plt

from Keke_PY.evaluation.evaluation import get_performance_graph_from_timeline, get_best_instance_on_batch_timeline, \
    get_all_performances_from_timeline
from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.HeuristicTreeRepresentation import HeuristicTreeRepresentation
from Keke_PY.heuristic_pymoo_representations.TrackedRepresentation import TrackedRepresentation
from Keke_PY.heuristic_pymoo_representations.WeightedHeuristicSumRepresentation import \
    WeightedHeuristicSumRepresentation


file_name_template: str = "/".join([
    "Keke_PY", "experiment_logs", "small_time_dependent_test_50gens_2sec",
    "TreeVsLinear_TimeBased_50Gens_2sec_JOB_TREES-out.txt"
])

linear_setups: List[Tuple[bool, bool, int]] = [
    (False, True, 585903),
]
tree_setups: List[Tuple[bool, bool, int]] = [
    (True, True, 585903),
]

def get_synopsis(
        trees: bool, tracked: bool, job_nr: int,
        select_best_by_batch: int = 0, show_performance_on_batch: int = -1,
) -> Tuple[List[Tuple[float, Dict[int, float]]], str]:
    file_name: str = file_name_template.replace("JOB", str(job_nr)).replace("TREES", str(int(trees)))
    print(f"reading in data for '{file_name}'")
    with open(file_name) as file:
        log_lines: List[str] = file.readlines()
    representation = HeuristicTreeRepresentation() if trees else WeightedHeuristicSumRepresentation()
    if tracked:
        representation = TrackedRepresentation(representation)
        representation.load_from_lines(log_lines)
    problem_data: KekeProblem = KekeProblem.from_log_lines(representation, log_lines, time_dependent_performance_function=True)
    timeline: List[Tuple[float, Tuple[int, int]]] = get_best_instance_on_batch_timeline(problem_data, select_best_by_batch)
    plot_data: List[Tuple[float, Dict[int, float]]] = get_all_performances_from_timeline(problem_data, timeline)
    individual: np.ndarray = problem_data.past_instances_by_gen_and_index[timeline[-1][1]]
    if tracked:
        return plot_data, representation.serialize_untracked(individual)
    else:
        return plot_data, representation.serialize(individual)


def stretch_from_points(
        data: List[Tuple[float, Dict[int, float]]],
        step_size: float = 10,
) -> List[Dict[int, float]]:
    last_data_point: Dict[int, float] = None
    res: List[Dict[int, float]] = []
    for time_step, data_point in data:
        while (len(res) + 1) * step_size < time_step:
            res.append(last_data_point)
        last_data_point = data_point
    res.append(last_data_point)
    return res

def combine_runs_to_single_number(
        stretched_runs: List[List[Dict[int, float]]],
        batch: int = -1,
        combine: Callable[[List[float]], float] =
            lambda values: sum(values) / len(values)
) -> List[float]:
    max_length: int = max(len(run) for run in stretched_runs)
    for run in stretched_runs:
        while len(run) < max_length:
            run.append(run[-1])
    res: List[float] = []
    for i in range(max_length):
        res.append(combine([run[i][batch] for run in stretched_runs]))
    return res


def plot_distr(
        ax, runs, batch, color
):
    for run in runs:
        ax.plot(*zip(*enumerate([performance[batch] for performance in run])), linewidth=1.0, color=color)
    ax.plot(*zip(*enumerate(combine_runs_to_single_number(runs, batch))), linewidth=3.0, color=color)

if __name__ == "__main__":
    linear_runs: List[List[Dict[int, float]]] = [
        stretch_from_points(get_synopsis(*setup)[0])
        for setup in linear_setups
    ]
    tree_runs: List[List[Dict[int, float]]] = [
        stretch_from_points(get_synopsis(*setup)[0])
        for setup in tree_setups
    ]

    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots()
    ax.set(
        xlim=(0, 50),
        ylim=(0, 4.0),
    )
    plot_distr(ax, linear_runs, -1, "blue")
    plot_distr(ax, tree_runs, -1, "green")
    plt.show()


    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots()
    ax.set(
        xlim=(0, 50),
        ylim=(0, 4.0),
    )
    plot_distr(ax, linear_runs, 0, "blue")
    plot_distr(ax, tree_runs, 0, "green")
    plt.show()