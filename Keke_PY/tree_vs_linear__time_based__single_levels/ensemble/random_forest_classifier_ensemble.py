import math
from random import shuffle
from typing import List, Optional, Tuple, Dict

import numpy as np
import pulp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.heuristics.hand_crafted_heuristics import named_heuristics
from Keke_PY.keke_game.keke import parse_map, make_level, GameState
from Keke_PY.keke_game.simulation import load_level_set

file_name: str = "Keke_PY/tree_vs_linear__time_based__single_levels/experiment_logs/combined_logs.txt"

training_levels: List[str] = [
    level["ascii"] for level in load_level_set("./json_levels/train_LEVELS.json")["levels"]
]
testing_levels: List[str] = [
    level["ascii"] for level in load_level_set("./json_levels/test_LEVELS.json")["levels"]
]
all_levels: List[str] = [*training_levels, *testing_levels]

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

times_by_genome_str_and_level_str: Dict[Tuple[str, str], Optional[float]] = dict([
    ((genome_str, test_level), time_matrix[instance_i][test_i])
    for instance_i, genome_str in enumerate(genome_list)
    for test_i, test_level in enumerate(all_levels)
])

linear_genome_strs: List[str] = genome_list[0::2]
tree_genome_strs: List[str] = genome_list[1::2]

levels: range = range(len(solving_matrix) // 2)

linear_solving_matrix: List[List[bool]] = [solving_matrix[2 * i] for i in levels]
tree_solving_matrix: List[List[bool]] = [solving_matrix[2 * i + 1] for i in levels]

use_trees: bool = True # TODO: document

solving_matrix: List[List[bool]] = tree_solving_matrix if use_trees else linear_solving_matrix
genome_strs: List[str] = tree_genome_strs if use_trees else linear_genome_strs
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





minimal_solving_set_genome_strs: List[str] = [genome_strs[index] for index in minimal_solving_set]


# build level features and ideal classifications

level_feature_names_and_params: List[Tuple[str, Tuple[float, ...]]] = [
    ("number_of_goal_objects", ()),
    ("number_of_player_objects", ()),
    ("connectivity", ()),
    ("number_of_auto_movers", ()),

    ("number_of_stuck_is_words", ()),
    ("number_of_stuck_prefixes", ()),
    ("number_of_stuck_suffixes", ()),
    ("number_of_stuck_important_suffixes", ()),

    ("number_of_killer_objects", ()),
    ("number_of_pushable_objects", ()),
    ("number_of_sinkable_objects", ()),
    ("number_of_stopped_objects", ()),
    ("player_killer_distance", (-1.0,)),

    ("distance_to_winnable_objects", (-1.0,)),
    ("distance_to_words", (-1.0,)),
    ("distance_to_pushable_objects", (-1.0,)),

    ("number_of_newly_created_rules", ()),

    ("goal_reachability", ())
]

named_level_feature_heuristics: Dict[str, Heuristic] = dict([
    (name, dict(named_heuristics)[name].with_parameters(*parameters))
    for name, parameters in level_feature_names_and_params
])

def get_feature_value(level_str: str, feature_heuristic: Heuristic) -> float:
    initial_level_state: GameState = make_level(parse_map(level_str))
    ctx: dict = {
        "initial GameState": initial_level_state,
        "initial rules": set(initial_level_state.rules)
    }
    return feature_heuristic.run(initial_level_state, ctx)

feature_by_level_and_name: Dict[Tuple[str, str], float] = dict([
    ((level_str, feature_name), get_feature_value(level_str, feature_heuristic))
    for level_str in all_levels for feature_name, feature_heuristic in named_level_feature_heuristics.items()
])

def get_best_genome_of_set_on_level_str(genome_str_set: List[str], level_str: str) -> str:
    genome_str_set = list(genome_str_set) # to make sure, this is actually a list and not a set o.e.
    times: List[Optional[float]] = [
        times_by_genome_str_and_level_str[(genome_str, level_str)]
        for genome_str in genome_str_set
    ]
    penalty: List[float] = [math.inf if time is None else time for time in times]
    return genome_str_set[penalty.index(min(*penalty))]

best_genome_of_minimal_solving_set_label_for_level_str: Dict[str, str] = dict([
    (level_str, get_best_genome_of_set_on_level_str(minimal_solving_set_genome_strs, level_str))
    for level_str in all_levels
])

def get_scikit_learn_data_for_level_strs(level_strs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    level_strs = list(level_strs) # to make sure, this is actually a list and not a set o.e.
    # {array-like, sparse matrix} of shape (n_samples, n_features)
    x_values: np.ndarray = np.array([
        [
            feature_by_level_and_name[(level_str, feature_name)]
            for feature_name, _ in level_feature_names_and_params
        ]
        for level_str in level_strs
    ])
    y_values: List[str] = [
        best_genome_of_minimal_solving_set_label_for_level_str[level_str]
        for level_str in level_strs
    ]
    #shuffle(y_values)
    return x_values, np.array(y_values)

train_x, train_y = get_scikit_learn_data_for_level_strs(training_levels)
test_x, y_test = get_scikit_learn_data_for_level_strs(testing_levels)

classifier: RandomForestClassifier = RandomForestClassifier()

classifier.fit(train_x, train_y)

y_pred = classifier.predict(test_x)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

solvable_test_level_count: int = len([
    test_level for test_level in testing_levels
    if any(
        times_by_genome_str_and_level_str[(genome_str, test_level)] is not None
        for genome_str in minimal_solving_set_genome_strs
    )
])
solved_test_level_count: int = len([
    test_level for i, test_level in enumerate(testing_levels)
    if times_by_genome_str_and_level_str[(y_pred[i], test_level)] is not None
])
print(f"{solved_test_level_count}/{solvable_test_level_count}/{len(testing_levels)}")
#print(y_pred)


