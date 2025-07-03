import multiprocessing
from itertools import chain
from typing import List, Tuple, Dict, Optional, Iterable

import numpy
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import DuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize

from Keke_PY.heuristic_pymoo_representations.HeuristicRepresentation import HeuristicRepresentation
from Keke_PY.heuristic_pymoo_representations.HeuristicTreeRepresentation import HeuristicTreeRepresentation
from Keke_PY.heuristic_pymoo_representations.WeightedHeuristicSumRepresentation import \
    WeightedHeuristicSumRepresentation
from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.keke_game.keke import GameState
from Keke_PY.search_agents.HeuristicGuidedSearch import HeuristicGuidedSearch
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.ClassifierEnsemble import ClassifierEnsemble
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.VirtualEnsembleProblem import VirtualEnsembleProblem
from Keke_PY.tree_vs_linear__time_based__single_levels.ensemble.minimal_solving_set import get_minimal_solving_genomes
from Keke_PY.tree_vs_linear__time_based__single_levels.experiment_logs.read_in_combined_log import \
    times_by_genome_and_level, all_levels, training_levels


class HeuristicClassifierEnsemble(ClassifierEnsemble):

    indicators_and_heuristic_genomes: List[Tuple[Heuristic, str]]

    def __init__(
            self,
            indicators_and_heuristic_genomes: List[Tuple[Heuristic, str]],
            representation: HeuristicRepresentation
    ):
        super().__init__(ClassifierEnsemble.precompute_heuristics(
            [genome for _, genome in indicators_and_heuristic_genomes],
            representation
        ))
        self.indicators_and_heuristic_genomes = indicators_and_heuristic_genomes


    def get_heuristic_genome(self, initial_game_state: GameState) -> str:
        indicator_values: List[float] = [
            indicator.run(initial_game_state, indicator.get_ctx(initial_game_state))
            for indicator, _ in self.indicators_and_heuristic_genomes
        ]
        return self.indicators_and_heuristic_genomes[indicator_values.index(max(indicator_values))][1]

    @staticmethod
    def trained(
            indicator_representation: HeuristicRepresentation,
            ensemble_part_representation: HeuristicRepresentation,
            available_policies: List[str],
            training_levels: List[str],
            time_limit: float = 60.0,
            pop_size: int = 10,
            generation_count: int = 50,
    ) -> Tuple['HeuristicClassifierEnsemble', 'HeuristicClassifierEnsembleRepresentation']:
        ensemble_representation: HeuristicClassifierEnsembleRepresentation = HeuristicClassifierEnsembleRepresentation(
            indicator_representation,
            available_policies,
            ensemble_part_representation,
        )
        training_problem: VirtualEnsembleProblem = VirtualEnsembleProblem.default_problem(
            representation=ensemble_representation,
            executor=multiprocessing.Pool(20),
            max_node_expansions=None,
            max_calculation_time=time_limit,
            time_dependent_performance_function=True,
            agent_factory=HeuristicGuidedSearch.GuidedSearchFactory(),
            test_levels_or_src=[],
            training_levels_or_src=training_levels,
            logging_prefix="VIRTUAL_CLASSIFIER_TRAINING__"
        )
        minimize(
            training_problem,
            GA(pop_size=pop_size, **ensemble_representation.algorithm_arguments()),
            termination=("n_eval", pop_size * generation_count),
            verbose=True
        )

        best_individual_index: Tuple[int, int] = \
            max(training_problem.get_best_past_individuals_generation_nrs_and_indices(0))
        best_individual_encoded: np.ndarray = training_problem.past_instances_by_gen_and_index[best_individual_index]
        best_individual: HeuristicClassifierEnsemble = ensemble_representation.into_heuristic(best_individual_encoded)


        return best_individual, ensemble_representation

class HeuristicClassifierEnsembleRepresentation(HeuristicRepresentation):
    _indicator_repr: HeuristicRepresentation
    _indicator_repr_size: int
    _nr_of_ensemble_parts: int
    _ensemble_part_labels: List[str]
    _ensemble_parts_representation: HeuristicRepresentation
    def __init__(
            self,
            inner_representation: HeuristicRepresentation,
            ensemble_part_labels: List[str],
            ensemble_parts_representation: HeuristicRepresentation,
    ):
        self._indicator_repr = inner_representation
        self._nr_of_ensemble_parts = len(ensemble_part_labels)
        self._indicator_repr_size = inner_representation.get_problem_data().n_var
        self._ensemble_part_labels = ensemble_part_labels
        self._ensemble_parts_representation = ensemble_parts_representation


    def get_problem_data(self) -> Problem:
        inner_problem_data: Problem = self._indicator_repr.get_problem_data()
        return Problem(
            n_var=self._nr_of_ensemble_parts * self._indicator_repr_size,
            xl=list(chain(inner_problem_data.xl for _ in range(self._nr_of_ensemble_parts))),
            xu=list(chain(inner_problem_data.xu for _ in range(self._nr_of_ensemble_parts))),
            vtype=inner_problem_data.vtype,
        )
    def into_heuristic(self, x: np.ndarray, n: Optional[int] = None) -> HeuristicClassifierEnsemble:
        return HeuristicClassifierEnsemble([
            (
                self._indicator_repr.into_heuristic(x[n * self._indicator_repr_size: (n + 1) * self._indicator_repr_size]),
                self._ensemble_part_labels[n]
            )
            for n in range(len(self._ensemble_part_labels))
        ], self._indicator_repr)
        assert n.__class__ == int, "This indicator_representation contains multiple heuristics, so it isn't clear, which one should be returned."
        assert 0 <= n < self._nr_of_ensemble_parts, f"n={n} should be in [0, {self._nr_of_ensemble_parts})"
        return self._indicator_repr.into_heuristic(x[n * self._indicator_repr_size: (n + 1) * self._indicator_repr_size])
    def into_indicators(self, x: np.ndarray) -> List[Heuristic]:
        return [
            self.into_heuristic(x, part)
            for part in range(self._nr_of_ensemble_parts)
        ]
    def serialize(self, x: np.ndarray) -> str:
        parts: List[str] = [
            self._indicator_repr.serialize(x[n * self._indicator_repr_size: (n + 1) * self._indicator_repr_size])
            for n in range(self._nr_of_ensemble_parts)
        ]
        lengths_of_parts: List[int] = [part.count(';') for part in parts]
        return f"{';'.join(map(str, lengths_of_parts))};{';'.join(parts)}"
    def deserialize(self, x: str) -> np.ndarray:
        entries: List[str] = x.split(';')
        lengths_of_parts: Iterable[int] = map(int, entries[:self._nr_of_ensemble_parts])
        inner_serialized: List[str] = []
        read_index: int = self._nr_of_ensemble_parts
        for part_length in lengths_of_parts:
            inner_serialized.append(';'.join(entries[ read_index : read_index+part_length ]))
            read_index += part_length
        inner_deserialized: List[np.ndarray] = [self._indicator_repr.deserialize(part) for part in inner_serialized]
        return np.array(chain(inner_deserialized))
    @property
    def sampling(self) -> Sampling:
        return HeuristicClassifierEnsembleRepresentation.TupleSampling(self)
    @property
    def mutation(self) -> Mutation:
        return HeuristicClassifierEnsembleRepresentation.TupleMutation(self)
    @property
    def crossover(self) -> Crossover:
        return HeuristicClassifierEnsembleRepresentation.TupleCrossover(self)
    @property
    def duplicate_elimination(self) -> DuplicateElimination:
        return HeuristicClassifierEnsembleRepresentation.TupleDuplicateElimination(self)


    def _get_inner_problem(self, problem: Problem, part: int) -> Problem:
        assert problem.n_var == self._indicator_repr_size * self._nr_of_ensemble_parts
        return Problem(
            n_var=self._indicator_repr_size,
            n_obj=problem.n_obj,
            n_ieq_constr=problem.n_ieq_constr,
            n_eq_constr=problem.n_eq_constr,
            xl=problem.xl[ part*self._indicator_repr_size: (part + 1) * self._indicator_repr_size],
            xu=problem.xu[ part*self._indicator_repr_size: (part + 1) * self._indicator_repr_size],
            vtype=problem.vtype,
        )

    class TupleSampling(Sampling):
        _ensemble_representation: 'HeuristicClassifierEnsembleRepresentation'
        _inner_sampling: Sampling

        def __init__(self, tuple_representation: 'HeuristicClassifierEnsembleRepresentation'):
            self._ensemble_representation = tuple_representation
            self._inner_sampling = tuple_representation._indicator_repr.sampling
            super().__init__()

        def _do(self, problem, n_samples, **kwargs) -> np.ndarray:
            inners: List[np.ndarray] = [
                self._inner_sampling._do(self._ensemble_representation._get_inner_problem(problem, part), n_samples, **kwargs)
                for part in range(self._ensemble_representation._nr_of_ensemble_parts)
            ]
            return numpy.concatenate(inners, 1)

    class TupleMutation(Mutation):
        _ensemble_repr: 'HeuristicClassifierEnsembleRepresentation'
        _inner_mutation: Mutation

        def __init__(self, tuple_representation: 'HeuristicClassifierEnsembleRepresentation'):
            self._ensemble_repr = tuple_representation
            self._inner_mutation = tuple_representation._indicator_repr.mutation
            super().__init__()
            self.prob = self._inner_mutation.prob.get()

        def do(self, problem, pop, inplace=True, **kwargs):
            self.prob = self._inner_mutation.prob.get()
            return Mutation.do(self, problem, pop, inplace, **kwargs)

        def _do(self, problem, x, **kwargs):
            inner_xs: List[np.ndarray] = [
                x[:, part*self._ensemble_repr._indicator_repr_size: (part + 1) * self._ensemble_repr._indicator_repr_size]
                for part in range(self._ensemble_repr._nr_of_ensemble_parts)
            ]
            inner_results: List[np.ndarray] = [
                self._inner_mutation._do(self._ensemble_repr._get_inner_problem(problem, part), inner_x, **kwargs)
                for part, inner_x in enumerate(inner_xs)
            ]
            if all(
                id(inner_results[i]) == id(inner_xs[i])
                for i in range(self._ensemble_repr._nr_of_ensemble_parts)
            ):
                return x
            return numpy.concatenate(inner_results, 1)

    class TupleCrossover(Crossover):
        _ensemble_repr: 'HeuristicClassifierEnsembleRepresentation'
        _inner_crossover: Crossover

        def __init__(self, tuple_representation: 'HeuristicClassifierEnsembleRepresentation'):
            self._ensemble_repr = tuple_representation
            self._inner_crossover = tuple_representation._indicator_repr.crossover
            super().__init__(
                self._inner_crossover.n_parents,
                self._inner_crossover.n_offsprings
            )
            self.prob = self._inner_crossover.prob.get()

        def do(self, problem, pop, parents=None, **kwargs):
            self.prob = self._inner_crossover.prob.get()
            return Crossover.do(self, problem, pop, parents, **kwargs)

        def _do(self, problem, x, **kwargs):
            inner_xs: List[np.ndarray] = [
                x[:, :, part*self._ensemble_repr._indicator_repr_size: (part + 1) * self._ensemble_repr._indicator_repr_size]
                for part in range(self._ensemble_repr._nr_of_ensemble_parts)
            ]
            inner_results: List[np.ndarray] = [
                self._inner_crossover._do(self._ensemble_repr._get_inner_problem(problem, part), inner_x, **kwargs)
                for part, inner_x in enumerate(inner_xs)
            ]
            return numpy.concatenate(inner_results, 2)

    class TupleDuplicateElimination(DuplicateElimination):
        _ensemble_repr: 'HeuristicClassifierEnsembleRepresentation'
        _inner_duplicate_elimination: DuplicateElimination
        def __init__(self, tuple_representation: 'HeuristicClassifierEnsembleRepresentation'):
            self._ensemble_repr = tuple_representation
            self._inner_duplicate_elimination = tuple_representation._indicator_repr.duplicate_elimination
            super().__init__()

        def _do(self, pop, other, is_duplicate):
            def inner_individual(individual: Individual, i: int) -> Individual:
                res = individual.copy()
                res.X = individual.X[i * self._ensemble_repr._indicator_repr_size: (i + 1) * self._ensemble_repr._indicator_repr_size]
                return res
            def inner_population(p: Population, i: int) -> Population:
                return Population([
                    inner_individual(individual, i)
                    for individual in p
                ])
            inner_is_duplicate = np.copy(is_duplicate)
            for part in range(self._ensemble_repr._nr_of_ensemble_parts):
                inner_pop: Population = inner_population(pop, part)
                inner_other = None if other is None else inner_population(other, part)
                self._inner_duplicate_elimination._do(inner_pop, inner_other, inner_is_duplicate)
                is_duplicate &= inner_is_duplicate
            return is_duplicate


    def __str__(self):
        return f"({HeuristicRepresentation.__str__(self)} of {self._nr_of_ensemble_parts} * {str(self._indicator_repr)})"




if __name__ == "__main__":

    use_trees: bool = True # TODO: make function argument!
    only_use_training_levels: bool = True # TODO: make function argument!


    minimal_solving_genomes: List[str] = get_minimal_solving_genomes(use_trees, only_use_training_levels, False)

    level_and_times_for_policy: List[Tuple[str, Dict[str, Optional[float]]]] = [
        (genome, dict((level, times_by_genome_and_level[(genome, level)]) for level in all_levels))
        for genome in minimal_solving_genomes
    ]

    representation: HeuristicRepresentation = HeuristicTreeRepresentation(3) if use_trees else WeightedHeuristicSumRepresentation()

    res, res_repr = HeuristicClassifierEnsemble.trained(
        representation,
        representation,
        minimal_solving_genomes,
        training_levels
    )

    # TODO: train ensemble and measure in actual runtime

