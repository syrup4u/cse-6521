from protocol import AbstractState

from itertools import product
import logging
import z3

logger = logging.getLogger(__name__)

class Z3Verifier:
    """
    Z3-based verifier for given protocol specifications and state transitions.
    """
    def __init__(self, state_class: AbstractState, num_nodes: int, num_rounds: int):
        # Avoid interference with other Z3 instances
        self.ctx = z3.Context()
        # Z3 core
        self.solver = z3.Solver(ctx=self.ctx)
        self.constraints = []
        # Z3 Variable Definition
        state_names = list(map(lambda x: x.name, state_class))
        self.valid_type, self.valid_states = z3.EnumSort('State', state_names, ctx=self.ctx)
        self.type_mapping = dict(zip(state_names, self.valid_states))
        self.lost_state = None
        # Environment Parameters
        self.num_nodes = num_nodes
        self.num_rounds = num_rounds
        # Transition Tracking
        self.all_states = [
            [z3.Const(f'state_r{r}_n{n}', self.valid_type) for n in range(num_nodes)] for r in range(num_rounds+1)
        ] # num_rounds x num_nodes
        self.all_messages = [
            [
                [z3.Const(f'message_r{r}_n{n}_from{f}', self.valid_type) for n in range(num_nodes)]
                for f in range(num_nodes)
            ]
            for r in range(num_rounds)
        ] # num_rounds x num_nodes x num_nodes
        # Initialize
        self._state_constraints(state_class)
        self._transition_constraints()
        self._message_constraints()

    def _state_constraints(self, state_class: AbstractState):
        lost_states = [self.type_mapping[state_class.get_lost_state().name]]
        initial_states = [self.type_mapping[s.name] for s in state_class.get_initial_states()]
        final_states = [self.type_mapping[s.name] for s in state_class.get_final_states()]
        # middle_states = [s for s in self.valid_states if s not in initial_states + final_states + lost_states]
        self.lost_state = lost_states[0]
        # S1: all states must be valid in corresponding rounds
        init_c = [z3.Or([self.all_states[0][n] == init_s for init_s in initial_states]) for n in range(self.num_nodes)]
        # mid_c = [
        #     z3.Or([self.all_states[r][n] == mid_s for mid_s in middle_states+lost_states])
        #     for r in range(1, self.num_rounds - 1)
        #     for n in range(self.num_nodes)
        # ]
        final_c = [
            z3.Or(
                [self.all_states[-1][n] == final_s for final_s in final_states+lost_states]
            ) for n in range(self.num_nodes)
        ]
        # S2: almost (num_rounds-1) nodes can be lost in total
        s2_c = [
            z3.PbLe(
                [
                    (z3.Or(
                        [
                            self.all_messages[r][n][f] == self.lost_state \
                            for r in range(self.num_rounds) \
                                for n in range(self.num_nodes)
                        ]), 1)
                    for f in range(self.num_nodes)
                ],
                self.num_rounds - 1
            )
        ]
        # S3: active nodes can make decision at the final round even their messages are lost
        s3_c = [
            z3.Implies(
                self.all_states[self.num_rounds-1][n] != self.lost_state,
                self.all_states[self.num_rounds][n] != self.lost_state
            ) for n in range(self.num_nodes)
        ]
        self.constraints += init_c + final_c + s2_c + s3_c

    def _transition_constraints(self):
        # T1: once a node is in lost state, it remains lost
        t1_c = [
            z3.Implies(
                self.all_states[r][n] == self.lost_state,
                self.all_states[r+1][n] == self.lost_state
            ) for r in range(1, self.num_rounds) for n in range(self.num_nodes)
        ]
        # T2: once a node's message is marked lost, this node is lost in the next round,
        # unless it is the last round
        t2_c = [
            z3.Implies(
                self.all_messages[r][n][f] == self.lost_state,
                z3.If(
                    r + 1 < self.num_rounds,
                    self.all_states[r+1][f] == self.lost_state,
                    True
                )
            ) for r in range(self.num_rounds) for n in range(self.num_nodes) for f in range(self.num_nodes)
        ]
        self.constraints += t1_c + t2_c

    def _message_constraints(self):
        # M1: messages should be either from corresponding states or lost
        # Note that message from itself cannot be lost
        m1_c = [
            z3.If(
                n == f,
                self.all_messages[r][n][f] == self.all_states[r][f],
                z3.Or(
                    self.all_messages[r][n][f] == self.all_states[r][f],
                    self.all_messages[r][n][f] == self.lost_state
                )
            ) for f in range(self.num_nodes) for r in range(self.num_rounds) for n in range(self.num_nodes)
        ]
        self.constraints += m1_c
    
    def add_state_constraints_test(self, states: list[AbstractState], round_number):
        """
        (for test)
        Add constraints for specific states at a given round.

        round_number: 0 means initial state before round-0
        """
        state_c = [
            self.all_states[round_number][n] == self.type_mapping[states[n].name]
            for n in range(self.num_nodes)
        ]
        self.constraints += state_c
    
    def add_constraint(self, custom_constraint):
        self.constraints.append(custom_constraint)

    def verify(self):
        self.solver.add(self.constraints)
        res = self.solver.check()
        if res == z3.sat:
            logger.info("Constraints are satisfiable.")
            m = self.solver.model()
            self.print_states(m)
            self.print_messages(m)
        elif res == z3.unsat:
            logger.info("Constraints are unsatisfiable.")
        else:
            logger.info("Solver returned unknown result.")
    
    def print_states(self, model):
        logger.info("===== States per Round =====")
        for r in range(self.num_rounds + 1):
            states_in_round = [model.evaluate(self.all_states[r][n]) for n in range(self.num_nodes)]
            logger.info(f"Round {r}: " + ", ".join([str(s) for s in states_in_round]))
        logger.info("============================")

    def print_messages(self, model):
        logger.info("===== Messages per Round =====")
        for r in range(self.num_rounds):
            logger.info(f"Round {r} Messages:")
            for n in range(self.num_nodes):
                messages_from_n = [model.evaluate(self.all_messages[r][n][f]) for f in range(self.num_nodes)]
                logger.info(f"  For Node {n}: " + ", ".join([str(m) for m in messages_from_n]))
        logger.info("==============================")
