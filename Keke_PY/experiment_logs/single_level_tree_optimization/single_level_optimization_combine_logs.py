import multiprocessing
import pathlib
from typing import List, Tuple, Optional

from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.HeuristicTreeRepresentation import HeuristicTreeRepresentation
from Keke_PY.heuristic_pymoo_representations.TrackedRepresentation import TrackedRepresentation
from Keke_PY.keke_game.simulation import load_level_set

log_location: str = "Keke_PY/experiment_logs/single_level_tree_optimization/long_100_gens_single_training_logs"

levels: List[str] = [
    *[level["ascii"] for level in
      load_level_set("./json_levels/train_LEVELS.json")["levels"]],
    *[level["ascii"] for level in
      load_level_set("./json_levels/test_LEVELS.json")["levels"]],
]

def get_level_results(level_nr: int) -> KekeProblem:
    slurm_job_id: int = 553991 # 523700 if level_nr != 183 else 524904
    file_name: str = f"keke_long_single_level_training_{slurm_job_id}_{level_nr}-out.txt"
    with open(pathlib.Path(log_location, file_name)) as file:
        lines: [str] = file.readlines()
    representation: TrackedRepresentation = TrackedRepresentation(
        HeuristicTreeRepresentation()
    )
    representation.load_from_lines(lines)
    split_index: int = lines.index("-----!!!NEW PROBLEM!!!-----\n")
    training_lines, testing_lines = lines[:split_index], lines[split_index:]
    training_data: KekeProblem = KekeProblem.from_log_lines(representation, training_lines)
    assert training_data.training_batches[0][0] == levels[level_nr]
    testing_data: KekeProblem = KekeProblem.from_log_lines(representation, testing_lines)
    print(f"reading in level {level_nr + 1} of {len(levels)} done.")
    return testing_data

def get_evaluation_str(level_nr: int) -> List[str]:
    testing_data: KekeProblem = get_level_results(level_nr)
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
        get_evaluation_str, range(len(levels))
    )

    print("\n---EVALUATION---\n")
    for evaluations in evaluations_list:
        print("LEVEL AGENT EVALUATION:" + '\t:'.join(evaluations))
