from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pulp
from pymoo.core.problem import Problem

from Keke_PY.experiments.KekeProblem import KekeProblem
from Keke_PY.heuristic_pymoo_representations.HeuristicRepresentation import HeuristicRepresentation
from Keke_PY.heuristic_pymoo_representations.HeuristicTreeRepresentation import HeuristicTreeRepresentation
from Keke_PY.heuristic_pymoo_representations.TupleRepresentation import TupleRepresentation
from Keke_PY.heuristic_pymoo_representations.WeightedHeuristicSumRepresentation import \
    WeightedHeuristicSumRepresentation
from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.keke_game.keke import GameState, make_level, parse_map
from Keke_PY.keke_game.simulation import load_level_set
from Keke_PY.search_agents.HeuristicGuidedSearch import HeuristicGuidedSearch

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






# Train Ensemble
"""
def get_first_heuristic_value(serialized_heuristic: str, level_nr: int) -> float:
    representation: HeuristicRepresentation = HeuristicTreeRepresentation() if use_trees else WeightedHeuristicSumRepresentation()
    heuristic: Heuristic = representation.into_heuristic(representation.deserialize(serialized_heuristic))
    initial_state: GameState = make_level(parse_map(all_levels[level_nr]))
    ctx = {
        "initial GameState": initial_state,
        "initial rules": set(initial_state.rules)
    }
    return heuristic.run(initial_state, ctx)
"""

class EnsembleProblem(KekeProblem):

    ensemble_parts: int
    tuple_representation: TupleRepresentation
    inner_representation: HeuristicRepresentation
    level_and_times_for_policy: List[Tuple[str, Dict[str, Optional[float]]]]

    def __init__(
            self,
            representation: HeuristicRepresentation,
            level_and_times_for_policy: List[Tuple[str, Dict[str, Optional[float]]]],
    ):
        self.inner_representation = representation
        self.ensemble_parts = len(level_and_times_for_policy)
        self.tuple_representation = TupleRepresentation(representation, self.ensemble_parts)
        self.level_and_times_for_policy = level_and_times_for_policy
        super().__init__(
            training_batches = [list(level_and_times_for_policy[0][1].keys())],
            representation = self.tuple_representation,
            max_node_expansions = None,
            max_calculation_time = 60.0,
            time_dependent_performance_function = False,
            executor = ProcessPoolExecutor(),
            test_batch = [],
            agent_factory = HeuristicGuidedSearch.GuidedSearchFactory(),
            silent = False,
            logging_prefix = "",
        )

    def register_and_run_next_generation(self, instances: [np.ndarray]):

        if self.is_continuation(instances):
            self.log_line("RESTART FROM PREVIOUS GENERATION DETECTED => SIMULATION WILL BE SKIPPED; RESULTS FROM PREVIOUS SIMULATION ARE RETURNED TO CALLER")
            self.try_continue_from_last_generation = False
            return

        self.log_generation_data(list(instances))
        self.past_instances_by_gen_and_index.update(((self.generation, index), deepcopy(instance)) for index, instance in enumerate(instances))

        self.fake_run_as_next_generation([
            self.tuple_representation.into_heuristics(instance)
            for instance in instances
        ])

    def fake_run_as_next_generation(self, instances: List[List[Heuristic]]):
        simulation_results: Dict[Tuple[int, str], Tuple[Union[List[str], None], int, float]] = {}
        for level_str in self.all_levels:
            initial_level_state: GameState = make_level(parse_map(level_str))
            ctx = {
                "initial GameState": initial_level_state,
                "initial rules": set(initial_level_state.rules)
            }
            for i, ensemble_parts in enumerate(instances):
                priorities: List[float] = [
                    classifier_part.run(initial_level_state, ctx)
                    for classifier_part in ensemble_parts
                ]
                chosen_policy: int = priorities.index(max(priorities))
                stored_solving_time: Optional[float] = self.level_and_times_for_policy[chosen_policy][1][level_str]
                fake_solution: Optional[List[str]] = None if stored_solving_time is None else ["NOT STORED"] # TODO: log real solution
                fake_node_expansions: int = None # TODO: log real node_expansions
                simulation_results[(i, level_str)] = (fake_solution, fake_node_expansions, stored_solving_time)

        self.past_evaluations_by_gen_index_and_level_id.update(((self.generation, key[0], self.level_to_id_map[key[1]]), result) for key, result in simulation_results.items())
        self.log_simulation_data(simulation_results)

        self.generation += 1



level_and_times_for_policy: List[Tuple[str, Dict[str, Optional[float]]]] = []


for line in lines:
    if line.startswith("LEVEL AGENT EVALUATION"):
        _, *results = line.split(':')
        solving_matrix.append([result.strip() != "----" for result in results])
        genome: str = genome_list[len(level_and_times_for_policy)]
        times: List[Optional[float]] = [None if result.strip() == "----" else float(result) for result in results]
        assert len(times) == len(all_levels)
        level_and_times_for_policy.append((genome, dict(zip(all_levels, times))))


representation: HeuristicRepresentation = HeuristicTreeRepresentation(3) if use_trees else WeightedHeuristicSumRepresentation()

ensemble_problem: EnsembleProblem = EnsembleProblem(representation, level_and_times_for_policy)

