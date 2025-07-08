import os
from typing import List, Tuple

import numpy as np

from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.DummyRepresentation import DummyRepresentation
from Keke_PY.heuristic_pymoo_representations.HeuristicRepresentation import HeuristicRepresentation
from Keke_PY.heuristic_pymoo_representations.HeuristicTreeRepresentation import HeuristicTreeRepresentation
from Keke_PY.heuristic_pymoo_representations.TrackedRepresentation import TrackedRepresentation
from Keke_PY.heuristic_pymoo_representations.WeightedHeuristicSumRepresentation import \
    WeightedHeuristicSumRepresentation
from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic

file_name_template: str = os.path.join(
    "Keke_PY",
    "tree_vs_linear__time_based",
    "experiment_logs",
    "TreeVsLinear_TimeBased_50Gens_60sec_%%_%-out.txt"
)
slurm_job_numbers: List[str] = [
    "859535", "859537", "859539", "859540", "859541",
    "859543", "859544", "859545", "859546", "859547",
]


def file_lines(index: int, trees: bool) -> List[str]:
    file_name: str = (
        file_name_template
            .replace("%%", slurm_job_numbers[index])
            .replace("%", '1' if trees else '0')
    )
    with open(file_name) as file:
        lines: List[str] = file.readlines()
    return lines




def get_level_results(index: int, trees: bool) -> (KekeProblem, HeuristicRepresentation, np.ndarray):
    file_name: str = (
        file_name_template
        .replace("%%", slurm_job_numbers[index])
        .replace("%", '1' if trees else '0')
    )
    representation: HeuristicRepresentation = TrackedRepresentation(
        HeuristicTreeRepresentation() if trees else WeightedHeuristicSumRepresentation()
    )
    with open(file_name) as file:
        lines: List[str] = file.readlines()

    # reading in training data:
    representation.load_from_lines(lines, accepting_prefixes=("ACTUAL_TRAINING__",))
    split_index: int = lines.index("-----!!!NEW PROBLEM!!!-----\n")
    training_lines, testing_lines = lines[:split_index], lines[split_index:]
    training_data: KekeProblem = KekeProblem.from_log_lines(representation, training_lines, logging_prefix="ACTUAL_TRAINING__")
    #assert training_data.training_batches[0][0] == levels[level_nr]

    # extracting the best individual:
    best_individual_index: Tuple[int, int] = max(training_data.get_best_past_individuals_generation_nrs_and_indices(0))
    best_individual_encoded: np.ndarray = training_data.past_instances_by_gen_and_index[best_individual_index]
    best_individual: Heuristic = representation.into_heuristic(best_individual_encoded)
    test_best_individual_representation: HeuristicRepresentation = DummyRepresentation(best_individual)

    # reading in testing data:
    #   the logging_prefix="BASELINE__" should ideally be  logging_prefix="TESTING__"
    #       the mistake is in the logs because of a mistake in
    #           'Keke_PY/tree_vs_linear__time_based/experiments/training.py'
    testing_data: KekeProblem = KekeProblem.from_log_lines(test_best_individual_representation, testing_lines, logging_prefix="BASELINE__")
    return testing_data, representation._inner_repr, best_individual_encoded[representation._offset:]


def extract_best_heuristic(index: int, trees: bool) -> Heuristic:
    _testing_data, representation, genome = get_level_results(index, trees)
    return representation.into_heuristic(genome)


if __name__ == "__main__":
    pass