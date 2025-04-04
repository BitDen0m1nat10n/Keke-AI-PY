import itertools
import math
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from copy import deepcopy
from itertools import chain
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
from pymoo.core.problem import Problem

from Keke_PY.keke_game.keke import GameState, make_level, parse_map
from Keke_PY.heuristic_pymoo_representations.HeuristicRepresentation import HeuristicRepresentation
from Keke_PY.keke_game.simulation import load_level_set
from Keke_PY.search_agents.HeuristicGuidedSearch import HeuristicGuidedSearch
from Keke_PY.search_agents.ai_interface import AgentFromPolicy, AIInterface


class KekeProblem(Problem):

    training_batches: List[List[str]]
    test_batch: List[str]
    all_levels: List[str]

    max_node_expansions: Optional[int]
    max_calculation_time: float
    time_dependent_performance_function: bool

    executor: Executor

    representation: HeuristicRepresentation
    agent_factory: AgentFromPolicy

    generation: int
    past_instances_by_gen_and_index: Dict[Tuple[int, int], np.ndarray]
    past_evaluations_by_gen_index_and_level_id: Dict[Tuple[int, int, int], Tuple[Union[List[str], None], int, float]]

    logging_prefix: Optional[str]


    def __init__(
            self,
            training_batches: List[List[str]],
            representation: HeuristicRepresentation,
            max_node_expansions: Optional[int] = 2000,
            max_calculation_time: float = math.inf,
            time_dependent_performance_function: bool = False,
            executor: Executor = ProcessPoolExecutor(),
            test_batch: List[str] = (),
            agent_factory: AgentFromPolicy = HeuristicGuidedSearch.GuidedSearchFactory(),
            silent: bool = False,
            logging_prefix: Optional[str] = ""
    ):
        self.generation = 0
        self.past_instances_by_gen_and_index = {}
        self.past_evaluations_by_gen_index_and_level_id = {}
        self.training_batches = training_batches
        self.test_batch = test_batch
        self.max_node_expansions = max_node_expansions
        self.max_calculation_time = max_calculation_time
        self.time_dependent_performance_function = time_dependent_performance_function
        training_levels = set(chain(*training_batches))
        assert all(level not in training_levels for level in test_batch), "Training on test-levels is not allowed!"
        self.all_levels = sorted(list(training_levels)) + test_batch
        self.level_to_id_map: Dict[str, int] = dict(map(lambda t: (t[1], t[0]), enumerate(self.all_levels)))
        self.representation = representation
        self.executor = executor
        self.agent_factory = agent_factory
        self.logging_prefix = logging_prefix

        problem_data: Problem = representation.get_problem_data()
        super().__init__(
            n_var = problem_data.n_var,
            n_obj = len(training_batches),
            n_ieq_constr = 0,
            xl=problem_data.xl,
            xu=problem_data.xu,
            vtype=problem_data.vtype,
        )
        if not silent:
            self.log_level_data()

    @classmethod
    def default_problem(
            cls,
            representation: HeuristicRepresentation,
            executor: Executor = ProcessPoolExecutor(),
            agent_factory: AgentFromPolicy = HeuristicGuidedSearch.GuidedSearchFactory(),
            training_levels_or_src: Union[List[str], str] = "./json_levels/train_LEVELS.json",
            test_levels_or_src: Union[List[str], str] = "./json_levels/test_LEVELS.json",
            limit_levels: int = None,
            max_calculation_time: float = math.inf,
            max_node_expansions: Optional[int] = 2000,
            time_dependent_performance_function: bool = False,
            logging_prefix: Optional[str] = ""
    ):
        if training_levels_or_src.__class__ == str:
            training_levels_or_src = [
                level["ascii"]
                for level in load_level_set(training_levels_or_src)["levels"]
            ]
        training_levels: List[str] = training_levels_or_src
        if test_levels_or_src.__class__ == str:
            test_levels_or_src = [
                level["ascii"]
                for level in load_level_set(test_levels_or_src)["levels"]
            ]
        test_levels: List[str] = test_levels_or_src
        if limit_levels is not None:
            training_levels = training_levels[:limit_levels]
            test_levels = test_levels[:limit_levels]
        return cls(
            training_batches=[training_levels],
            representation=representation,
            executor=executor,
            test_batch=test_levels,
            agent_factory=agent_factory,
            max_calculation_time=max_calculation_time,
            max_node_expansions=max_node_expansions,
            time_dependent_performance_function=time_dependent_performance_function,
            logging_prefix=logging_prefix,
        )

    def run_as_next_generation(self, instances: [AIInterface]):
        simulation_data_list: List[Tuple[Tuple[int, AIInterface], str, Optional[int], float]] = list(itertools.product(
            enumerate(instances),
            self.all_levels,
            [self.max_node_expansions],
            [self.max_calculation_time],
        ))
        simulation_results: Dict[Tuple[int, str], Tuple[Union[List[str], None], int, float]] = dict(list(self.executor.map(
            evaluate_ai_on_level, simulation_data_list
        )))
        self.past_evaluations_by_gen_index_and_level_id.update(((self.generation, key[0], self.level_to_id_map[key[1]]), result) for key, result in simulation_results.items())
        self.log_simulation_data(simulation_results)

        self.generation += 1

    def register_and_run_next_generation(self, instances: [np.ndarray]):

        self.log_generation_data(list(instances))
        self.past_instances_by_gen_and_index.update(((self.generation, index), deepcopy(instance)) for index, instance in enumerate(instances))

        self.run_as_next_generation(
            map(lambda arr: self.agent_factory.make_agent_from_policy(self.representation.into_heuristic(arr)), instances)
        )

    def _evaluate(self, x, out, *args, **kwargs):
        assert all(len(batch) > 0 for batch in self.training_batches), "Be aware, that there is no training without training data! (call register_and_run_next_generation instead)"

        self.register_and_run_next_generation(x)

        performance_of_instance_on_batch: Dict[Tuple[int, int], float] = self.get_performance_of_instance_on_batch()

        # this output is supposed to be minimized:
        out["F"] = np.zeros((len(x), len(self.training_batches)))

        for batch_nr in range(len(self.training_batches)):
            for agent_nr in range(len(x)):
                out["F"][agent_nr, batch_nr] = -performance_of_instance_on_batch[(agent_nr, batch_nr)]

        # There are no constrains:
        out["G"] = np.zeros((len(x), 0))


    def get_performance_of_instance_on_batch(
            self,
            generation: int = -1,
            override_time_dependence_for_fitness: bool = None,
    ) -> Dict[Tuple[int, int], float]:
        if override_time_dependence_for_fitness is None:
            override_time_dependence_for_fitness = self.time_dependent_performance_function
        if generation == -1:
            generation = self.generation - 1

        max_value: float = self.max_calculation_time if override_time_dependence_for_fitness else self.max_node_expansions
        value_index: int = 2 if override_time_dependence_for_fitness else 1

        nr_of_instances: int = max(
            key[1]
            for key in self.past_evaluations_by_gen_index_and_level_id.keys()
            if key[0] == generation
        ) + 1

        performance_of_instance_on_batch: Dict[Tuple[int, int], float] = {}

        for batch_nr, batch in itertools.chain([(-1, self.test_batch)], enumerate(self.training_batches)):
            nr_of_levels: int = len(batch)
            for agent_nr in range(nr_of_instances):
                if nr_of_levels == 0:
                    performance_of_instance_on_batch[(agent_nr, batch_nr)] = math.nan
                    continue
                value_sum: float = sum(
                    self.past_evaluations_by_gen_index_and_level_id[
                        (generation, agent_nr, self.level_to_id_map[level])
                    ][value_index]
                    for level in batch
                )
                average_value: float = value_sum / nr_of_levels
                average_leftover_value: float = max_value - average_value
                nr_of_solved_levels: int = sum(
                    self.past_evaluations_by_gen_index_and_level_id[
                        (generation, agent_nr, self.level_to_id_map[level])
                    ][0] is not None
                    for level in batch
                )
                ratio_of_solved_levels: float = nr_of_solved_levels / nr_of_levels
                solved_level_bonus: float = ratio_of_solved_levels * max_value
                performance_of_instance_on_batch[(agent_nr, batch_nr)] = average_leftover_value + solved_level_bonus

        return performance_of_instance_on_batch

    def get_performances_of_all_generations_instances_and_batches(
            self,
            override_time_dependence_for_fitness: bool = None,
    ) -> Dict[Tuple[int, int, int], float]:
        res: Dict[Tuple[int, int, int], float] = {}
        for generation in range(self.generation):
            res.update(
                ((generation, index, batch), performance)
                for (index, batch), performance in self.get_performance_of_instance_on_batch(generation, override_time_dependence_for_fitness).items()
            )
        return res

    def get_best_past_individuals_generation_nrs_and_indices(
            self,
            batch: int = 0,
            override_time_dependence_for_fitness: bool = None,
    ) -> List[Tuple[int, int]]:
        performances: Dict[Tuple[int, int, int], float] = self.get_performances_of_all_generations_instances_and_batches(override_time_dependence_for_fitness)
        best_instances: List[Tuple[int, int]] = []
        best_performance: float = -math.inf
        for (generation, index, batch_nr), performance in performances.items():
            if batch_nr == batch:
                if performance > best_performance:
                    best_instances = [(generation, index)]
                    best_performance = performance
                elif performance == best_performance:
                    best_instances.append((generation, index))
        return best_instances

    def total_evaluation_time_per_generation(self) -> List[float]:
        res: List[float] = [0.0] * self.generation
        for (gen, _, _), (_, _, search_time) in self.past_evaluations_by_gen_index_and_level_id.items():
            res[gen] += search_time
        return res



    def log_line(self, line: str):
        if self.logging_prefix is None:
            return
        # TODO: the following line should be done by the caller
        line = line.replace('\\','\\\\').replace('\n', '\\n')
        assert len(line.split('\n')) == 1
        print(self.logging_prefix + line)


    def log_level_data(self):
        for level_id, level in enumerate(self.all_levels):
            self.log_line(f"LEVEL:{level_id}:{level}")
        for batch_id, batch in chain(enumerate(self.training_batches), [(-1, self.test_batch)]):
            for index, level in enumerate(batch):
                level_id = self.level_to_id_map[level]
                assert self.all_levels[level_id] == level
                self.log_line(f"BATCH:{batch_id}:{index}:{level_id}")
    
    @classmethod
    def from_log_lines(
            cls,
            representation: HeuristicRepresentation,
            lines: List[str],
            max_node_expansions: int = 2000,
            executor: Executor = None,
            agent_factory: AgentFromPolicy = HeuristicGuidedSearch.GuidedSearchFactory(),
            max_calculation_time: float = 2.0,
            time_dependent_performance_function: bool = False,
            logging_prefix: str = "",
            new_logging_prefix: Optional[str] = 0 # wrong type => copy from logging_prefix [since None is valid type]
    ):
        if new_logging_prefix.__class__ != str and new_logging_prefix is not None:
            new_logging_prefix = logging_prefix
        all_levels: Dict[int, str] = {}
        batches: List[Tuple[int, int, int]] = []
        for line in lines:
            if line.startswith(logging_prefix + "LEVEL:"):
                _, level_id, level = line.split(':')
                all_levels[int(level_id)] = level.strip().replace('\\n', '\n')
            elif line.startswith(logging_prefix + "BATCH:"):
                _, batch_id, index, level_id = line.split(':')
                batches.append((int(batch_id), int(index), int(level_id)))
        training_batches: List[List[str]] = []
        test_batch: List[str] = []
        batches.sort()
        for batch_id, index, level_id in batches:
            while batch_id >= len(training_batches):
                training_batches.append([])
            batch: List[str] = test_batch if batch_id == -1 else training_batches[batch_id]
            assert index == len(batch)
            batch.append(all_levels[level_id])
        res: KekeProblem = cls(
            training_batches, representation,
            max_node_expansions, max_calculation_time,
            time_dependent_performance_function,
            executor, test_batch, agent_factory,
            True, new_logging_prefix
        )
        for line in lines:
            if line.startswith(logging_prefix + "EVAL_INSTANCE:"):
                _, generation, index, serialized_instance = line.split(':')
                res.past_instances_by_gen_and_index.update(
                    [((int(generation), int(index)), representation.deserialize(serialized_instance))])
                res.generation = max(res.generation, int(generation) + 1)
            if line.startswith(logging_prefix + "RUN_RESULT:"):
                _, generation, index, old_level_id, *result = line.split(':')
                new_level_id: int = res.level_to_id_map[all_levels[int(old_level_id)]]
                solution: Union[List[str], None]
                node_expansions: int
                calculation_time: float
                assert result is not None, f"Unexpected value in : {line}"
                # the following cases are the types of logging, for this code to be compatible with old logs
                if len(result) == 1:
                    # logging only node_expansions / "----" for no solution
                    result: str = result[0].strip()
                    if result == "----":
                        node_expansions = max_node_expansions
                        solution = None
                    else:
                        assert result.isdigit(), f"Unexpected value in '{result}'"
                        node_expansions = int(result)
                        solution = ["solution was not logged"]
                    calculation_time = math.nan
                elif len(result) == 2:
                    # logging node_expansions and solution/"----"
                    str_node_expansions, str_solution = result
                    str_solution = str_solution.strip()
                    assert str_node_expansions.isdigit(), f"Unexpected value in : {line}"
                    node_expansions = int(str_node_expansions)
                    if str_solution == "----":
                        solution = None
                    else:
                        solution = list(iter(str_solution))
                    calculation_time = math.nan
                else:
                    # currently used logging:
                    assert len(result) == 3, f"Unexpected value in : {line}"
                    # logging node_expansions, calculation_time and solution/"----"
                    str_node_expansions, str_calculation_time, str_solution = result
                    str_solution = str_solution.strip()
                    assert str_node_expansions.isdigit(), f"Unexpected value in : {line}"
                    node_expansions = int(str_node_expansions)
                    if str_solution == "----":
                        solution = None
                    else:
                        solution = list(iter(str_solution))
                    calculation_time = float(str_calculation_time)
                result: Tuple[Union[List[str], None], int, float] = (solution, node_expansions, calculation_time)
                res.past_evaluations_by_gen_index_and_level_id.update([(
                    (int(generation), int(index), new_level_id),
                    result
                )])
        return res

    def log_generation_data(self, x: list):
        for index, instance in enumerate(x):
            self.log_line(f"EVAL_INSTANCE:{self.generation}:{index}:{self.representation.serialize(instance)}")

    def log_simulation_data(self, simulation_results: Dict[Tuple[int, str], Tuple[Union[List[str], None], int, float]]):
        for (index, level), (solution, forward_model_calls, calc_time) in simulation_results.items():
            solution_str: str = '----' if solution is None else ''.join(sol[0] for sol in solution)
            self.log_line(f"RUN_RESULT:{self.generation}:{index}:{self.level_to_id_map[level]}:{forward_model_calls}:{calc_time}:{solution_str}")

    def log_performances(self, performance_of_instance_on_batch: Dict[Tuple[int, int], float]):
        nr_of_instances: int = max(key[0] for key in performance_of_instance_on_batch.keys()) + 1
        for batch_nr, batch in itertools.chain([(-1, self.test_batch)], enumerate(self.training_batches)):
            performances_on_batch: str = "\t|\t".join(
                str(performance_of_instance_on_batch[(instance, batch_nr)]) for instance in range(nr_of_instances)
            )
            self.log_line(f"PERFORMANCE:{self.generation}:{batch_nr}:\t{performances_on_batch}")


def evaluate_ai_on_level(
    simulation_data: Tuple[Tuple[int, AIInterface], str, Optional[int], float]
) -> Tuple[Tuple[int, str], Tuple[Union[List[str], None], int, float]]:
    ai_index: int = simulation_data[0][0]
    agent: AIInterface = simulation_data[0][1]
    level: str = simulation_data[1]
    max_forward_model_calls: Optional[int] = simulation_data[2]
    max_calculation_time: float = simulation_data[3]
    start_state: GameState = make_level(parse_map(level))
    start_time: float = time.time()
    solution: Tuple[Union[List[str], None], int] = agent.search(
        start_state,
        max_forward_model_calls,
        None,
        max_calculation_time,
        False
    )
    end_time: float = time.time()
    #print((ai_index, level), solution[0], solution[0], solution[1], end_time - start_time)
    #print(end_time - start_time, solution[1], solution[0])
    return (ai_index, level), (solution[0], solution[1], end_time - start_time)
