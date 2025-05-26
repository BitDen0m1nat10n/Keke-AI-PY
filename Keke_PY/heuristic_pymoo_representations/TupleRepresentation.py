from itertools import chain
from typing import List, Optional, Iterable

import numpy
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import DuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling

from Keke_PY.heuristic_pymoo_representations.HeuristicRepresentation import HeuristicRepresentation
from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic


class TupleRepresentation(HeuristicRepresentation):
    _inner_repr: HeuristicRepresentation
    _inner_repr_size: int
    _inner_instances_per_tuple_instance: int
    def __init__(self, inner_representation: HeuristicRepresentation, inner_instances_per_tuple_instance: int):
        self._inner_repr = inner_representation
        self._inner_instances_per_tuple_instance = inner_instances_per_tuple_instance
        self._inner_repr_size = inner_representation.get_problem_data().n_var


    def get_problem_data(self) -> Problem:
        inner_problem_data: Problem = self._inner_repr.get_problem_data()
        return Problem(
            n_var=self._inner_instances_per_tuple_instance * self._inner_repr_size,
            xl=chain([inner_problem_data.xl] * self._inner_instances_per_tuple_instance),
            xu=chain([inner_problem_data.xu] * self._inner_instances_per_tuple_instance),
            vtype=inner_problem_data.vtype,
        )
    def into_heuristic(self, x: np.ndarray, n: Optional[int] = None) -> Heuristic:
        assert n.__class__ == int, "This representation contains multiple heuristics, so it isn't clear, which one should be returned."
        assert 0 <= n < self._inner_instances_per_tuple_instance, f"n={n} should be in [0, {self._inner_instances_per_tuple_instance})"
        return self._inner_repr.into_heuristic(x[ n*self._inner_repr_size : (n+1)*self._inner_repr_size ])
    def into_heuristics(self, x: np.ndarray) -> List[Heuristic]:
        return [
            self.into_heuristic(x, part)
            for part in range(self._inner_instances_per_tuple_instance)
        ]
    def serialize(self, x: np.ndarray) -> str:
        parts: List[str] = [
            self._inner_repr.serialize(x[ n*self._inner_repr_size : (n+1)*self._inner_repr_size ])
            for n in range(self._inner_instances_per_tuple_instance)
        ]
        lengths_of_parts: List[int] = [part.count(';') for part in parts]
        return f"{';'.join(map(str, lengths_of_parts))};{';'.join(parts)}"
    def deserialize(self, x: str) -> np.ndarray:
        entries: List[str] = x.split(';')
        lengths_of_parts: Iterable[int] = map(int, entries[:self._inner_instances_per_tuple_instance])
        inner_serialized: List[str] = []
        read_index: int = self._inner_instances_per_tuple_instance
        for part_length in lengths_of_parts:
            inner_serialized.append(';'.join(entries[ read_index : read_index+part_length ]))
            read_index += part_length
        inner_deserialized: List[np.ndarray] = [self._inner_repr.deserialize(part) for part in inner_serialized]
        return np.array(chain(inner_deserialized))
    @property
    def sampling(self) -> Sampling:
        return TupleRepresentation.TupleSampling(self)
    @property
    def mutation(self) -> Mutation:
        return TupleRepresentation.TupleMutation(self)
    @property
    def crossover(self) -> Crossover:
        return TupleRepresentation.TupleCrossover(self)
    @property
    def duplicate_elimination(self) -> DuplicateElimination:
        return TupleRepresentation.TupleDuplicateElimination(self)


    def _get_inner_problem(self, problem: Problem, part: int) -> Problem:
        assert problem.n_var == self._inner_repr_size * self._inner_instances_per_tuple_instance
        return Problem(
            n_var=self._inner_repr_size,
            n_obj=problem.n_obj,
            n_ieq_constr=problem.n_ieq_constr,
            n_eq_constr=problem.n_eq_constr,
            xl=problem.xl[ part*self._inner_repr_size : (part+1)*self._inner_repr_size ],
            xu=problem.xu[ part*self._inner_repr_size : (part+1)*self._inner_repr_size ],
            vtype=problem.vtype,
        )

    class TupleSampling(Sampling):
        _tuple_representation: 'TupleRepresentation'
        _inner_sampling: Sampling

        def __init__(self, tuple_representation: 'TupleRepresentation'):
            self._tuple_representation = tuple_representation
            self._inner_sampling = tuple_representation._inner_repr.sampling
            super().__init__()

        def _do(self, problem, n_samples, **kwargs) -> np.ndarray:
            inners: List[np.ndarray] = [
                self._inner_sampling._do(self._tuple_representation._get_inner_problem(problem, part), n_samples, **kwargs)
                for part in range(self._tuple_representation._inner_instances_per_tuple_instance)
            ]
            return numpy.concatenate(inners, 1)

    class TupleMutation(Mutation):
        _tuple_repr: 'TupleRepresentation'
        _inner_mutation: Mutation

        def __init__(self, tuple_representation: 'TupleRepresentation'):
            self._tuple_repr = tuple_representation
            self._inner_mutation = tuple_representation._inner_repr.mutation
            super().__init__()
            self.prob = self._inner_mutation.prob.get()

        def do(self, problem, pop, inplace=True, **kwargs):
            self.prob = self._inner_mutation.prob.get()
            return Mutation.do(self, problem, pop, inplace, **kwargs)

        def _do(self, problem, x, **kwargs):
            inner_xs: List[np.ndarray] = [
                x[:, part*self._tuple_repr._inner_repr_size : (part+1)*self._tuple_repr._inner_repr_size ]
                for part in range(self._tuple_repr._inner_instances_per_tuple_instance)
            ]
            inner_results: List[np.ndarray] = [
                self._inner_mutation._do(self._tuple_repr._get_inner_problem(problem, part), inner_x, **kwargs)
                for part, inner_x in enumerate(inner_xs)
            ]
            if all(
                id(inner_results[i]) == id(inner_xs[i])
                for i in range(self._tuple_repr._inner_instances_per_tuple_instance)
            ):
                return x
            return numpy.concatenate(inner_results, 1)

    class TupleCrossover(Crossover):
        _tuple_repr: 'TupleRepresentation'
        _inner_crossover: Crossover

        def __init__(self, tuple_representation: 'TupleRepresentation'):
            self._tuple_repr = tuple_representation
            self._inner_crossover = tuple_representation._inner_repr.crossover
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
                x[:, :, part*self._tuple_repr._inner_repr_size : (part+1)*self._tuple_repr._inner_repr_size ]
                for part in range(self._tuple_repr._inner_instances_per_tuple_instance)
            ]
            inner_results: List[np.ndarray] = [
                self._inner_crossover._do(self._tuple_repr._get_inner_problem(problem, part), inner_x, **kwargs)
                for part, inner_x in enumerate(inner_xs)
            ]
            return numpy.concatenate(inner_results, 2)

    class TupleDuplicateElimination(DuplicateElimination):
        _tuple_repr: 'TupleRepresentation'
        _inner_duplicate_elimination: DuplicateElimination
        def __init__(self, tuple_representation: 'TupleRepresentation'):
            self._tuple_repr = tuple_representation
            self._inner_duplicate_elimination = tuple_representation._inner_repr.duplicate_elimination
            super().__init__()

        def _do(self, pop, other, is_duplicate):
            def inner_individual(individual: Individual, i: int) -> Individual:
                res = individual.copy()
                res.X = individual.X[i * self._tuple_repr._inner_repr_size: (i + 1) * self._tuple_repr._inner_repr_size]
                return res
            def inner_population(p: Population, i: int) -> Population:
                return Population([
                    inner_individual(individual, i)
                    for individual in p
                ])
            inner_is_duplicate = np.copy(is_duplicate)
            for part in range(self._tuple_repr._inner_instances_per_tuple_instance):
                inner_pop: Population = inner_population(pop, part)
                inner_other = None if other is None else inner_population(other, part)
                self._inner_duplicate_elimination._do(inner_pop, inner_other, inner_is_duplicate)
                is_duplicate &= inner_is_duplicate
            return is_duplicate


    def __str__(self):
        return f"({HeuristicRepresentation.__str__(self)} of {self._inner_instances_per_tuple_instance} * {str(self._inner_repr)})"