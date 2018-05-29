from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects

            Action(Load(c, p, a),
            	PRECOND: At(c, a) ∧ At(p, a) ∧ Cargo(c) ∧ Plane(p) ∧ Airport(a)
            	EFFECT: ¬ At(c, a) ∧ In(c, p))
            """
            # all load ground actions
            loads = []

            for c in self.cargos:
                for p in self.planes:
                    for a in self.airports:

                        # list of preconditions
                        precond_pos = [expr("At({}, {})".format(c, a)),
                                       expr("At({}, {})".format(p, a)),]
                        precond_neg = []

                        # list of effects
                        effect_add = [expr("In({}, {})".format(c, p)),]
                        effect_rem = [expr("At({}, {})".format(c, a)),]

                        # load actions of cargo, planes & airports
                        load = Action(expr("Load({}, {}, {})".format(c, p, a)),
                                     [precond_pos, precond_neg],
                                     [effect_add, effect_rem])

                        # append load actions to list of Action objects
                        loads.append(load)

            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects

            Action(Unload(c, p, a),
            	PRECOND: In(c, p) ∧ At(p, a) ∧ Cargo(c) ∧ Plane(p) ∧ Airport(a)
            	EFFECT: At(c, a) ∧ ¬ In(c, p))
            """
            # all unload ground actions
            unloads = []

            for c in self.cargos:
                for p in self.planes:
                    for a in self.airports:

                        # list of preconditions
                        precond_pos = [expr("In({}, {})".format(c, p)),
                                       expr("At({}, {})".format(p, a)),]
                        precond_neg = []

                        # list of effects
                        effect_add = [expr("In({}, {})".format(c, a)),]
                        effect_rem = [expr("At({}, {})".format(c, p)),]

                        # unload actions of cargo, planes & airports
                        load = Action(expr("Unload({}, {}, {})".format(c, p, a)),
                                     [precond_pos, precond_neg],
                                     [effect_add, effect_rem])

                        # append unload actions to list of Action objects
                        loads.append(load)

            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects

            Action(Fly(p, from, to),
            	PRECOND: At(p, from) ∧ Plane(p) ∧ Airport(from) ∧ Airport(to)
            	EFFECT: ¬ At(p, from) ∧ At(p, to))
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # possible actions
        possible_actions = []

        # knowledge base; tell sentence of state
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())

        # loop through action_list to check with pos & neg preconditions
        for action in self.actions_list:
            is_legal = True # assume legal action

            # not legal action if positive preconditions is not in clauses
            for clause in action.precond_pos:
                is_legal = clause not in kb.clauses

            # not legal action if negative precondition is in clauses
            for clause in action.precond_neg:
                is_legal = clause in kb.clauses

            # append to list of actions if is_legal
            possible_actions.append(action) if is_legal

        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        new_state = FluentState([], [])

        # get old state
        old_state = decode_state(state, self.state_map)

        # new_state.pos from the result of old_state.pos & action.effect_add
        new_state.pos.extend(
        [action if action not in action.effect_rem for action in old_state.pos] +
        [action if action not in new_state.pos for action in action.effect_add])

        # new_state.neg from the result of old_state.neg & action.effect_rem
        new_state.neg.extend(
        [action if action not in action.effect_add for action in old_state.neg] +
        [action if action not in new_state.neg for action in action.effect_rem])

        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        count = 0
        return count


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    # TODO implement Problem 2 definition
    pass


def air_cargo_p3() -> AirCargoProblem:
    # TODO implement Problem 3 definition
    pass
