import math
from typing import Dict, List, Tuple, Optional

from Keke_PY.experiments.KekeProblem import KekeProblem


def get_best_instance_on_batch_timeline(
        data: KekeProblem,
        batch: 0,
        override_time_dependence_for_fitness: bool = None,
        evaluations_as_time_and_not_total_evaluation_time: bool = True,
) -> List[Tuple[float, Tuple[int, int]]]:
    time_per_generation: List[float] = () if evaluations_as_time_and_not_total_evaluation_time else (
        data.total_evaluation_time_per_generation()
    )
    performances: Dict[Tuple[int, int, int], float] = (
        data.get_performances_of_all_generations_instances_and_batches(override_time_dependence_for_fitness)
    )
    current_individual: Optional[Tuple[int, int]] = None
    current_performance: float = -math.inf
    cumulative_time_or_evaluations: float = 0.0
    res: List[Tuple[float, Tuple[int, int]]] = []
    for generation in range(data.generation):
        generation_size: int = max(
            individual
            for gen, individual, _ in performances.keys()
            if gen == generation
        ) + 1
        cumulative_time_or_evaluations += generation_size if evaluations_as_time_and_not_total_evaluation_time else (
            time_per_generation[generation]
        )
        best_individual_changed: bool = False
        for individual in range(generation_size):
            if performances[(generation, individual, batch)] > current_performance:
                current_individual = (generation, individual)
                current_performance = performances[(generation, individual, batch)]
                best_individual_changed = True
        if best_individual_changed:
            res.append((cumulative_time_or_evaluations, current_individual))
    return res

def get_performance_graph_from_timeline(
        data: KekeProblem,
        timeline: List[Tuple[float, Tuple[int, int]]],
        batch: int = -1,
        override_time_dependence_for_fitness: bool = None,
) -> List[Tuple[float, float]]:
    performances: Dict[Tuple[int, int, int], float] = (
        data.get_performances_of_all_generations_instances_and_batches(override_time_dependence_for_fitness)
    )
    return [
        (past_used_time, performances[(gen, i, batch)])
        for past_used_time, (gen, i) in timeline
    ]


def get_all_performances_from_timeline(
        data: KekeProblem,
        timeline: List[Tuple[float, Tuple[int, int]]],
        override_time_dependence_for_fitness: bool = None,
) -> List[Tuple[float, Dict[int, float]]]:
    performances: Dict[Tuple[int, int, int], float] = (
        data.get_performances_of_all_generations_instances_and_batches(override_time_dependence_for_fitness)
    )
    return [
        (past_used_time, dict((batch, performances[(gen, i, batch)]) for batch in range(-1, len(data.training_batches))))
        for past_used_time, (gen, i) in timeline
    ]