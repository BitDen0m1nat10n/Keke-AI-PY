from typing import List, Optional

import pulp
from Keke_PY.keke_game.simulation import load_level_set

file_name: str = "Keke_PY/tree_vs_linear__time_based__single_levels/experiment_logs/combined_logs.txt"

all_levels: List[str] = [
    *[level["ascii"] for level in
      load_level_set("./json_levels/train_LEVELS.json")["levels"]],
    *[level["ascii"] for level in
      load_level_set("./json_levels/test_LEVELS.json")["levels"]],
]

with open(file_name) as file:
    lines: List[str] = file.readlines()


time_matrix: List[List[Optional[float]]] = []
solving_matrix: List[List[bool]] = []
genome_list: List[str] = []

for line in lines:
    if line.startswith("LEVEL AGENT EVALUATION"):
        _, *results = line.split(':')
        time_matrix.append([None if result.strip() == "----" else float(result) for result in results])
        solving_matrix.append([result.strip() != "----" for result in results])
    if line.startswith("LEVEL AGENT GENOME"):
        _, *result = line.split(':')
        genome_list.append(':'.join(result))

levels: range = range(len(solving_matrix) // 2)

linear_solving_matrix: List[List[bool]] = [solving_matrix[2 * i] for i in levels]
tree_solving_matrix: List[List[bool]] = [solving_matrix[2 * i + 1] for i in levels]

use_trees: bool = True # TODO: document

solving_matrix: List[List[bool]] = tree_solving_matrix if use_trees else linear_solving_matrix
training_level_count: int = len(load_level_set("./json_levels/train_LEVELS.json")["levels"])


solvable_levels: List[bool] = [any(heur[lvl] for heur in solving_matrix) for lvl in levels]

reduce_to_training_levels: bool = True # TODO: document
if reduce_to_training_levels:
    solving_matrix = solving_matrix[:training_level_count]
    levels: range = range(training_level_count)








# Get Minimal-Solving-Set
# The following code is a modified version of https://github.com/AlbrErik/bachelor-thesis/blob/main/KekeCompetition-main/OptimizingKekeAgents/evaluation/min_heu_module.py

problem = pulp.LpProblem("Minimal_Solving_Set", pulp.LpMinimize)

use_heuristic: List[pulp.LpVariable] = [
    pulp.LpVariable(f'h_{i}', cat='Binary')
    for i in range(len(solving_matrix))
]
solved_levels: List[pulp.LpVariable] = [
    pulp.LpVariable(f'l_{i}', cat='Binary')
    for i in levels
]

problem += pulp.lpSum(use_heuristic)

for lvl in levels:
    if not any(heur[lvl] for heur in solving_matrix):
        continue
    problem += pulp.lpSum(
        heur_used
        for i, heur_used in enumerate(use_heuristic)
        if solving_matrix[i][lvl]
    ) >= solved_levels[lvl]

problem += pulp.lpSum(solved_levels) >= len(solving_matrix)

print(f"constraints:{len(problem.constraints)}")
problem.solve()

minimal_solving_set: List[int] = [heur for heur in range(len(solving_matrix)) if pulp.value(use_heuristic[heur]) == 1]
print(minimal_solving_set)

all_solved_levels: List[bool] = [any(solving_matrix[e][lvl] for e in minimal_solving_set) for lvl in range(len(solving_matrix[0]))]

for heur in minimal_solving_set:
    print(f'heur{heur}\t:\t' + ''.join('1' if solving_matrix[heur][lvl] else '0' for lvl in range(len(solving_matrix[0]))))
    genome_index: int = 2 * heur + (1 if use_trees else 0)
    print(genome_list[genome_index])

print('solved  :\t' + ''.join('1' if all_solved_levels[lvl] else '0' for lvl in range(len(solving_matrix[0]))))
print('solvable:\t' + ''.join('1' if solvable_levels[lvl] else '0' for lvl in range(len(solving_matrix[0]))))

print(f"{sum(all_solved_levels)}/{len(solving_matrix[0])} levels solved")




