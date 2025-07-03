import itertools
from typing import Tuple, List

from Keke_PY.tree_vs_linear__time_based__single_levels.experiment_logs.read_in_combined_log import \
    times_by_genome_and_level, level_for_genome, get_genomes, get_levels


def get_solved_level_fraction(
        level_set: List[str],
        genomes: List[str]
) -> Tuple[int, int]:
    solved: List[bool] = [
        any(times_by_genome_and_level[(genome, level)] is not None for genome in genomes)
        for level in level_set
    ]

    return sum(solved), len(level_set)


def successful_genome_fraction(
        genomes: List[str]
) -> Tuple[int, int]:
    return sum(
        times_by_genome_and_level[(genome, level_for_genome[genome])] is not None
        for genome in genomes
    ), len(genomes)

def frac_as_str(fraction: Tuple[int, int]) -> str:
    return f"{fraction[0]}/{fraction[1]}"

bool_values: List[bool] = [False, True]

if __name__ == "__main__":

    for use_trees, test_training_set, test_testing_set in itertools.product(bool_values, bool_values, bool_values):
        test_set_name: str
        if test_training_set:
            if test_testing_set:
                test_set_name = "full_set"
            else:
                test_set_name = "train_set"
        else:
            if test_testing_set:
                test_set_name = "test_set"
            else:
                continue
        pol_repr_name: str = "trees" if use_trees else "linear"
        genomes: List[str] = get_genomes(use_trees, test_training_set, test_testing_set)
        fract: Tuple[int, int] = successful_genome_fraction(genomes)
        print(f"{pol_repr_name} on {test_set_name} solve the level they are trained on {frac_as_str(fract)} times.")

    print("")

    for use_trees, use_test_set, test_training_set, test_testing_set in itertools.product(bool_values, bool_values, bool_values, bool_values):
        pol_repr_name: str = "trees" if use_trees else "linear"
        test_set_name: str
        if test_training_set:
            if test_testing_set:
                test_set_name = "full_set"
            else:
                test_set_name = "train_set"
        else:
            if test_testing_set:
                test_set_name = "test_set"
            else:
                continue
        train_set_name: str = "full_set" if use_test_set else "train_set"
        usable_genomes: List[str] = get_genomes(use_trees, True, use_test_set)
        levels_to_test_on: List[str] = get_levels(test_training_set, test_testing_set)
        fract: Tuple[int, int] = get_solved_level_fraction(levels_to_test_on, usable_genomes)
        print(f"{pol_repr_name} trained on {train_set_name}[{len(usable_genomes)}lvls] solve {frac_as_str(fract)} Levels of {test_set_name}.")
