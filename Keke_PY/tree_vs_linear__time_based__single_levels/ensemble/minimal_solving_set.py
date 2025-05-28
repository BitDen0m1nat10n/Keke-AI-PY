# The following code is a modified version of https://github.com/AlbrErik/bachelor-thesis/blob/main/KekeCompetition-main/OptimizingKekeAgents/evaluation/min_heu_module.py


from typing import List

import pulp

from Keke_PY.tree_vs_linear__time_based__single_levels.experiment_logs.read_in_combined_log import \
    training_levels, all_levels, times_by_genome_and_level, get_genomes


def get_minimal_solving_genomes(
        use_trees: bool,
        only_training_levels: bool = True,
        verbose: bool = False,
) -> List[str]:
    # Get Minimal-Solving-Set

    all_level_genomes: List[str] = get_genomes(use_trees)
    available_genome_set: List[str] = get_genomes(use_trees, True, not only_training_levels)

    required_level_set: List[str]
    if only_training_levels:
        required_level_set = training_levels
    else:
        required_level_set = all_levels


    problem = pulp.LpProblem("Minimal_Solving_Set", pulp.LpMinimize)

    use_heuristic: List[pulp.LpVariable] = [
        pulp.LpVariable(f'h_{genome}', cat='Binary')
        for genome in available_genome_set
    ]
    solved_levels: List[pulp.LpVariable] = [
        pulp.LpVariable(f'l_{lvl}', cat='Binary')
        for lvl in required_level_set
    ]

    problem += pulp.lpSum(use_heuristic)

    for lvl_nr, lvl in enumerate(required_level_set):
        if not any(times_by_genome_and_level[(genome, lvl)] is not None for genome in available_genome_set):
            continue
        problem += pulp.lpSum(
            heur_used
            for genome_nr, heur_used in enumerate(use_heuristic)
            if times_by_genome_and_level[(available_genome_set[genome_nr], lvl)] is not None
        ) >= solved_levels[lvl_nr]

    problem += pulp.lpSum(solved_levels) >= len(available_genome_set) # is this correct???

    if verbose:
        print(f"constraints:{len(problem.constraints)}")
    problem.solve()

    minimal_solving_set_indices: List[int] = [
        i for i in range(len(available_genome_set))
        if pulp.value(use_heuristic[i]) == 1
    ]
    minimal_solving_set: List[str] = [available_genome_set[i] for i in minimal_solving_set_indices]

    if verbose:
        print(minimal_solving_set_indices)

        all_solved_levels: List[bool] = [
            any(times_by_genome_and_level[(genome, lvl)] for genome in minimal_solving_set)
            for lvl in all_levels
        ]

        for i, genome in enumerate(minimal_solving_set):
            print(f'heur{i}\t:\t' + ''.join(
                '1' if times_by_genome_and_level[(genome, lvl)] else '0' for lvl in all_levels
            ))
            print(genome)

        print('solved  :\t' + ''.join('1' if all_solved_levels[i] else '0' for i in range(len(all_levels))))
        print('solvable:\t' + ''.join('1' if any(
            times_by_genome_and_level[(genome, level)] is not None
            for genome in all_level_genomes
        ) else '0' for level in all_levels))

        print(f"{sum(all_solved_levels)}/{len(all_levels)} levels solved")

    return minimal_solving_set


if __name__ == "__main__":
    get_minimal_solving_genomes(True, True, True)