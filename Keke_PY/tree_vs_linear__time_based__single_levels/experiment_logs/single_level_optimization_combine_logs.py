import multiprocessing
import pathlib
from typing import List, Tuple, Optional

from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.HeuristicRepresentation import HeuristicRepresentation
from Keke_PY.heuristic_pymoo_representations.HeuristicTreeRepresentation import HeuristicTreeRepresentation
from Keke_PY.heuristic_pymoo_representations.TrackedRepresentation import TrackedRepresentation
from Keke_PY.heuristic_pymoo_representations.WeightedHeuristicSumRepresentation import \
    WeightedHeuristicSumRepresentation
from Keke_PY.keke_game.simulation import load_level_set

log_location: str = "Keke_PY/tree_vs_linear__time_based__single_levels/experiment_logs/full_single_level_logs"
log_file_name: str = "KekeTimeBasedSingleLevelOptimization50Gens_%A_%a-out.txt"

levels: List[str] = [
    *[level["ascii"] for level in
      load_level_set("./json_levels/train_LEVELS.json")["levels"]],
    *[level["ascii"] for level in
      load_level_set("./json_levels/test_LEVELS.json")["levels"]],
]

slurm_job_id: int = 553991 # TODO: set to correct job_id

def level_and_representation_from_job_arr_index(job_arr_index: int) -> (int, HeuristicRepresentation):
    use_trees: bool = (job_arr_index % 2) == 1
    return (
        job_arr_index // 2,
        TrackedRepresentation(
            HeuristicTreeRepresentation() if use_trees else WeightedHeuristicSumRepresentation()
        )
    )

def get_level_results(job_arr_index: int) -> KekeProblem:
    file_name: str = log_file_name.replace("%A", str(slurm_job_id)).replace("%a", str(job_arr_index))
    level_nr, representation = level_and_representation_from_job_arr_index(job_arr_index)
    with open(pathlib.Path(log_location, file_name)) as file:
        lines: [str] = file.readlines()
    representation.load_from_lines(lines)
    split_index: int = lines.index("-----!!!NEW PROBLEM!!!-----\n")
    training_lines, testing_lines = lines[:split_index], lines[split_index:]
    training_data: KekeProblem = KekeProblem.from_log_lines(representation, training_lines, logging_prefix="TRAIN__")
    assert training_data.training_batches[0][0] == levels[level_nr]
    testing_data: KekeProblem = KekeProblem.from_log_lines(representation, testing_lines, logging_prefix="RESULT_ON_ALL_LEVELS__")
    print(f"reading in job_arr_index {job_arr_index + 1} of {2 * len(levels)} done.")
    return testing_data

def get_evaluation_str(job_arr_index: int) -> List[str]:
    testing_data: KekeProblem = get_level_results(job_arr_index)
    evaluations: List[str] = []
    for lvl in levels:
        evaluation: Tuple[Optional[List[str]], int] = testing_data.past_evaluations_by_gen_index_and_level_id[
            (0, 0, testing_data.level_to_id_map[lvl])
        ]
        if evaluation[0] is None:
            evaluations.append("----")
        else:
            evaluations.append(str(evaluation[1]))
    return evaluations

if __name__ == '__main__':

    print("\n---READING IN DATA---\n")

    evaluations_list: List[List[str]] = multiprocessing.Pool(6).map(
        get_evaluation_str, range(2 * len(levels))
    )

    print("\n---EVALUATION---\n")
    for evaluations in evaluations_list:
        print("LEVEL AGENT EVALUATION:" + '\t:'.join(evaluations))
