from protocol import AbstractState, AtomicCommitState as ACState, PrimaryBackupState as PBState
import common

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
        middle_states = [s for s in self.valid_states if s not in initial_states + final_states + lost_states]
        self.lost_state = lost_states[0]
        # S1: all states must be valid in corresponding rounds
        init_c = [z3.Or([self.all_states[0][n] == init_s for init_s in initial_states]) for n in range(self.num_nodes)]
        mid_c = [
            z3.Or([self.all_states[r][n] == mid_s for mid_s in middle_states+lost_states])
            for r in range(1, self.num_rounds - 1)
            for n in range(self.num_nodes)
        ]
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
        self.constraints += init_c + mid_c + final_c + s2_c + s3_c

    def _transition_constraints(self):
        # T1: once a node is in lost state, it remains lost
        t1_c = [
            z3.Implies(
                self.all_states[r][n] == self.lost_state,
                self.all_states[r+1][n] == self.lost_state
            ) for r in range(1, self.num_rounds - 1) for n in range(self.num_nodes)
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

    def rule_constraints(self, protocol_name, counter_example=False):
        if protocol_name == "atomic_commit":
            target_rules = self._atomic_commit_properties()
        elif protocol_name == "primary_backup":
            target_rules = self._primary_backup_properties()
        if counter_example:
            self.constraints += [z3.Not(z3.And(target_rules))]
        else:
            self.constraints += target_rules

    def add_transition_constraints(self, transitions):
        """
        Add custom transition constraints based on provided transition rules.

        transitions: [[messages], next_state, round_number]
        """
        transition_c = []
        for r in range(self.num_rounds):
            for n in range(self.num_nodes):
                transition_conditions = []
                for transition in transitions:
                    if transition[2] != r:
                        continue
                    msg_conditions = [
                        self.all_messages[r][n][f] == self.type_mapping[transition[0][f].name]
                        for f in range(self.num_nodes)
                    ]
                    next_state_condition = self.all_states[r+1][n] == self.type_mapping[transition[1].name]
                    # matching condition
                    transition_conditions.append(z3.Implies(
                        z3.And(msg_conditions),
                        next_state_condition
                    ))
                # this node's transition constraint for round r
                transition_c.append(z3.And(transition_conditions))
        self.constraints += transition_c
    
    def add_state_constraints(self, states: list[AbstractState], round_number):
        """
        Add constraints for specific states at a given round (for test).

        round_number: 0 means initial state before round-0
        """
        state_c = [
            self.all_states[round_number][n] == self.type_mapping[states[n].name]
            for n in range(self.num_nodes)
        ]
        self.constraints += state_c

    def _atomic_commit_properties(self):
        lost_message = z3.Or(
            [
                self.all_messages[r][n][f] == self.lost_state \
                    for r in range(self.num_rounds) \
                        for n in range(self.num_nodes) \
                            for f in range(self.num_nodes)
            ]
        )
        # R1: if all initial states are local commit and no lost, then all alive nodes must be commit
        r1_c = z3.Implies(
            z3.Not(lost_message),
            z3.Implies(
                z3.And([self.all_states[0][n] == self.type_mapping[ACState.LocalCommit.name] for n in range(self.num_nodes)]),
                z3.And([self.all_states[-1][n] != self.type_mapping[ACState.Abort.name] for n in range(self.num_nodes)])
            )
        )
        # R2: if any initial state is local abort, then all alive nodes must be abort
        r2_c = z3.Implies(
            z3.Or([self.all_states[0][n] == self.type_mapping[ACState.LocalAbort.name] for n in range(self.num_nodes)]),
            z3.And([self.all_states[-1][n] != self.type_mapping[ACState.Commit.name] for n in range(self.num_nodes)])
        )
        # R3: no contradiction in final states
        r3_c = z3.Not(
            z3.And(
                z3.Or([self.all_states[-1][n] == self.type_mapping[ACState.Commit.name] for n in range(self.num_nodes)]),
                z3.Or([self.all_states[-1][n] == self.type_mapping[ACState.Abort.name] for n in range(self.num_nodes)])
            )
        )
        return [r1_c, r2_c, r3_c]

    def _primary_backup_properties(self):
        # R1: final states must have corresponding initial states
        r1_c1 = z3.Implies(
            z3.And([self.all_states[0][n] == self.type_mapping[PBState.LocalOne.name] for n in range(self.num_nodes)]),
            z3.And([self.all_states[-1][n] != self.type_mapping[PBState.Zero.name] for n in range(self.num_nodes)])
        )
        r1_c2 = z3.Implies(
            z3.And([self.all_states[0][n] == self.type_mapping[PBState.LocalZero.name] for n in range(self.num_nodes)]),
            z3.And([self.all_states[-1][n] != self.type_mapping[PBState.One.name] for n in range(self.num_nodes)])
        )
        # R2: no contradiction in final states
        r2_c = z3.Not(
            z3.And(
                z3.Or([self.all_states[-1][n] == self.type_mapping[PBState.One.name] for n in range(self.num_nodes)]),
                z3.Or([self.all_states[-1][n] == self.type_mapping[PBState.Zero.name] for n in range(self.num_nodes)])
            )
        )
        return [r1_c1, r1_c2, r2_c]

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
                logger.info(f"  From Node {n}: " + ", ".join([str(m) for m in messages_from_n]))
        logger.info("==============================")

def add_test_case(z3_verifier: Z3Verifier):
    """
    To find a satisfiable example, we can fix the transition and state to test the verifier.

    Use with `rule_constraints(counter_example=False)`
    """
    test_transition = [
        [ACState.LocalCommit, ACState.LocalCommit, ACState.LocalCommit],
        ACState.DoNothing_One,
        0
    ]
    test_states = [
        [ACState.DoNothing_One, ACState.DoNothing_One, ACState.Lost],
        1
    ]
    z3_verifier.add_transition_constraints([test_transition])
    z3_verifier.add_state_constraints(test_states[0], test_states[1])

def verify_protocol(state_class: AbstractState, get_actions: callable, num_nodes, num_rounds):
    """
    Verify the given protocol by generating all possible transitions.
    Only supports `get_actions(messages, is_last_round)` signature.

    state_class: The class representing the states of the protocol.
    get_actions: the function of the protocol that takes a list of messages and returns the next state.
    """
    transitions = []
    # All possible initial states
    initial_states = state_class.get_initial_states()
    initial_states.append(state_class.get_lost_state())
    all_initial_comb = list(product(initial_states, repeat=num_nodes))
    all_next_states = get_actions(all_initial_comb, 0 if num_rounds > 1 else 1)
    round_list = [0] * len(all_initial_comb)
    transitions.extend(list(zip(all_initial_comb, all_next_states, round_list)))
    # Intermediate states exist
    if num_rounds > 1:
        middle_states = [s for s in state_class if s not in state_class.get_initial_states() + state_class.get_final_states()]
        all_middle_comb = list(product(middle_states, repeat=num_nodes))
        # All possible middle states to middle states
        # TODO: If using one flag to represent: initial -> middle -> final, then no need to create (round_num-2) copies here
        if num_rounds > 2:
            all_next_states = get_actions(all_middle_comb, 0)
        for r in range(1, num_rounds - 1):
            round_list = [r] * len(all_middle_comb)
            transitions.extend(list(zip(all_middle_comb, all_next_states, round_list)))
        # All possible middle states to final states
        all_next_states = get_actions(all_middle_comb, 1)
        round_list = [num_rounds-1] * len(all_middle_comb)
        transitions.extend(list(zip(all_middle_comb, all_next_states, round_list)))
    return transitions

def verify_protocol_with_rn(state_class: AbstractState, get_actions: callable, num_nodes, num_rounds):
    """
    With round number version of get_actions.
    """
    transitions = []
    # All possible initial states
    initial_states = state_class.get_initial_states()
    initial_states.append(state_class.get_lost_state())
    all_initial_comb = list(product(initial_states, repeat=num_nodes))
    all_next_states = get_actions(all_initial_comb, 0)
    round_list = [0] * len(all_initial_comb)
    transitions.extend(list(zip(all_initial_comb, all_next_states, round_list)))
    # Intermediate states exist
    if num_rounds > 1:
        middle_states = [s for s in state_class if s not in state_class.get_initial_states() + state_class.get_final_states()]
        all_middle_comb = list(product(middle_states, repeat=num_nodes))
        for r in range(1, num_rounds):
            all_next_states = get_actions(all_middle_comb, r)
            round_list = [r] * len(all_middle_comb)
            transitions.extend(list(zip(all_middle_comb, all_next_states, round_list)))
    return transitions

def verify_gt_protocol():
    num_nodes = 4
    num_rounds = 2
    protocol_names = ["atomic_commit", "primary_backup"]
    for p in protocol_names:
        z3_verifier = Z3Verifier(common.PROTOCOL_TABLE[p]['state'], num_nodes, num_rounds)
        z3_verifier.rule_constraints(p, counter_example=True)
        ts = verify_protocol(common.PROTOCOL_TABLE[p]['state'], common.PROTOCOL_TABLE[p]['groundtruth_class'].get_actions, num_nodes, num_rounds)
        z3_verifier.add_transition_constraints(ts)
        z3_verifier.verify()

def verify_model_protocol(protocol_name: str, state_class: AbstractState, get_actions: callable, num_nodes, num_rounds, encode_rn: bool):
    z3_verifier = Z3Verifier(state_class, num_nodes, num_rounds)
    z3_verifier.rule_constraints(protocol_name, counter_example=True)
    if encode_rn:
        ts = verify_protocol_with_rn(state_class, get_actions, num_nodes, num_rounds)
    else:
        ts = verify_protocol(state_class, get_actions, num_nodes, num_rounds)
    z3_verifier.add_transition_constraints(ts)
    z3_verifier.verify()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    verify_gt_protocol()
