from typing import List

from Keke_PY.heuristics.ParametrisedHeuristic import Heuristic
from Keke_PY.keke_game.keke import GameObj, GameState
from Keke_PY.search_agents.HeuristicGuidedSearch import HeuristicGuidedSearch
from Keke_PY.search_agents.ai_interface import AgentFromPolicy

"""
This is a reconstruction of 'Keke_JS/agents/defaultfixed_AGENT.js'.
We assume, that the whole file implements a policy-based-search
where the policy is defined in line 188:
	return [(win_d+word_d+push_d)/3, new node([...])]
"""

class DefaultAgent(HeuristicGuidedSearch):

    def __init__(self):
        super().__init__(DefaultAgentPolicy())

    class DefaultAgentFactory(AgentFromPolicy):
        def make_agent_from_policy(self, _policy: Heuristic) -> 'DefaultAgent':
            return DefaultAgent()




def dist(a: GameObj, b: GameObj) -> float:
    """
    Translation of 'function dist' from line 215 in 'Keke_JS/agents/defaultfixed_AGENT.js'.
    // BASIC EUCLIDEAN DISTANCE FUNCTION FROM OBJECT A TO OBJECT B

    Manhattan-distance between two objects
    @param a is the first game-object
    @param b is the second game-object
    @return manhattan-distance between a and b
    """
    return abs(a.x - b.x) + abs(a.y - b.y)


def heuristic2(group1: List[GameObj], group2: List[GameObj]) -> float:
    """
    Translation of 'function heuristic2' from line 193 in 'Keke_JS/agents/defaultfixed_AGENT.js'.
    //FIND AVERAGE DISTANCE OF GROUP THAT IS CLOSEST TO ANOTHER OBJECT IN A DIFFERENT GROUP
    Slightly optimized by building the sum immediately instead of building a list,
        and checking the lists for elements first.

    Calculates the average Distance between two groups of objects or 10_000.

    @param group1 First group of objects.
    @param group2 Second group of objects.
    @return Average distance between the given groups of objects or 10_000.

    """
    if len(group1) * len(group2) == 0:
        return 10_000 # default from line 209 in 'Keke_JS/agents/defaultfixed_AGENT.js'
    distance_sum: float = 0
    distances_count: int = 0
    for obj1 in group1:
        for obj2 in group2:
            distances_count += 1
            distance_sum += dist(obj1, obj2)
    return distance_sum / distances_count


class DefaultAgentPolicy(Heuristic):
    """
    Translation of policy in line 188 at the end of 'function getNextState'
        in line 158 in 'Keke_JS/agents/defaultfixed_AGENT.js'.
    """

    def run(self, state: GameState, ctx: dict, *args: float) -> float:
        """
        @param state is the current game-state
        @param _ctx contextual Information for the heuristic (unused)
        @return average of average-distances from the player to winnables, words and pushables
        """
        win_d: float = heuristic2(state.players, state.winnables)
        word_d: float = heuristic2(state.players, state.words)
        push_d: float = heuristic2(state.players, state.pushables)
        return (win_d + word_d + push_d) / 3.0

