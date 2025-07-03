from typing import List, Optional, Tuple, Dict

from Keke_PY.keke_game.simulation import load_level_set

_file_name: str = "Keke_PY/tree_vs_linear__time_based__single_levels/experiment_logs/combined_logs.txt"

training_levels: List[str] = [
    level["ascii"] for level in load_level_set("./json_levels/train_LEVELS.json")["levels"]
]
testing_levels: List[str] = [
    level["ascii"] for level in load_level_set("./json_levels/test_LEVELS.json")["levels"]
]
all_levels: List[str] = [*training_levels, *testing_levels]

with open(_file_name) as file:
    _lines: List[str] = file.readlines()


_time_matrix: List[List[Optional[float]]] = []
_genome_list: List[str] = []

for line in _lines:
    if line.startswith("LEVEL AGENT EVALUATION"):
        _, *results = line.split(':')
        _time_matrix.append([None if result.strip() == "----" else float(result) for result in results])
    if line.startswith("LEVEL AGENT GENOME"):
        _, *result = line.split(':')
        _genome_list.append(':'.join(result))

times_by_genome_and_level: Dict[Tuple[str, str], Optional[float]] = dict([
    ((genome_str, test_level), _time_matrix[instance_i][test_i])
    for instance_i, genome_str in enumerate(_genome_list)
    for test_i, test_level in enumerate(all_levels)
])

linear_genome_strs: List[str] = _genome_list[0::2]
tree_genome_strs: List[str] = _genome_list[1::2]

linear_genome_strs_on_training_set: List[str] = linear_genome_strs[:len(training_levels)]
linear_genome_strs_on_testing_set: List[str] = linear_genome_strs[len(training_levels):]
tree_genome_strs_on_training_set: List[str] = tree_genome_strs[:len(training_levels)]
tree_genome_strs_on_testing_set: List[str] = tree_genome_strs[len(training_levels):]

def get_genomes(trees: bool, training_set: bool = True, testing_set: bool = True) -> List[str]:
    if trees:
        return (
            tree_genome_strs_on_training_set if training_set else []
        ) + (
            tree_genome_strs_on_testing_set if testing_set else []
        )
    else:
        return (
            linear_genome_strs_on_training_set if training_set else []
        ) + (
            linear_genome_strs_on_testing_set if testing_set else []
        )

def get_levels(training_set: bool = True, testing_set: bool = True) -> List[str]:
    return (
        training_levels if training_set else []
    ) + (
        testing_levels if testing_set else []
    )


level_for_genome: Dict[str, str] = dict([
    (genome, all_levels[i // 2])
    for i, genome in enumerate(_genome_list)
])

genome_for_level: Dict[str, str] = dict([
    (all_levels[i // 2], genome)
    for i, genome in enumerate(_genome_list)
])


if __name__ == "__main__":
    def min_or_60(params: List[float]) -> float:
        if params.__class__ == float:
            return params
        if len(params) == 0:
            return 60.0
        return min(params)
    minimal_times: List[float] = [
        min_or_60(list(
            times_by_genome_and_level[(genome, level)]
            for genome in tree_genome_strs_on_training_set
            if times_by_genome_and_level[(genome, level)] is not None
        ))
        for level in all_levels
    ]
    print(sum(minimal_times)/len(minimal_times))