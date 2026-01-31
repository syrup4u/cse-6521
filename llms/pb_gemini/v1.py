from protocol import AbstractState
from llms.minism import MiniSM

import logging

logger = logging.getLogger(__name__)

class ConsensusState(AbstractState):
    # Format: (Value, Round, IsInitial, IsFinal)
    # Using 10 rounds as a safe upper bound for the emulator
    LOCAL_ZERO = (0, 0, True, False)
    LOCAL_ONE  = (1, 0, True, False)
    
    # Intermediate States: S_{Round}_{Value}
    S1_0 = (0, 1, False, False); S1_1 = (1, 1, False, False)
    S2_0 = (0, 2, False, False); S2_1 = (1, 2, False, False)
    S3_0 = (0, 3, False, False); S3_1 = (1, 3, False, False)
    
    COMMIT_ZERO = (0, 99, False, True)
    COMMIT_ONE  = (1, 99, False, True)
    CRASHED     = (-1, -1, False, False)

    @property
    def is_initial(self) -> bool:
        return self.value[2]

    @property
    def is_final(self) -> bool:
        return self.value[3]

    @classmethod
    def get_lost_state(cls):
        return cls.CRASHED

    @classmethod
    def get_initial_states(cls):
        return [cls.LOCAL_ZERO, cls.LOCAL_ONE]

    @classmethod
    def get_final_states(cls):
        return [cls.COMMIT_ZERO, cls.COMMIT_ONE]

def transition(current_state: ConsensusState, received_states: list[ConsensusState], total_nodes: int) -> ConsensusState:
    """
    Transition function to be called by the simulator each round.
    """
    if current_state.is_final or current_state == ConsensusState.CRASHED:
        return current_state

    # 1. Collect all seen values (current + incoming)
    all_seen = [current_state] + received_states
    
    # 2. Determine if we've ever seen a '1' (or a state that saw '1')
    saw_one = any(s.value[0] == 1 for s in all_seen if s != ConsensusState.CRASHED)
    current_val = 1 if saw_one else 0
    
    # 3. Increment round
    current_round = current_state.value[1]
    next_round = current_round + 1
    
    # 4. Decide: If we have reached N rounds, we must commit to ensure agreement
    # In a synchronous system, f+1 rounds (N here) guarantees all live nodes see the '1'
    if next_round >= total_nodes:
        return ConsensusState.COMMIT_ONE if current_val == 1 else ConsensusState.COMMIT_ZERO
    
    # 5. Otherwise, move to next intermediate state
    state_name = f"S{next_round}_{current_val}"
    return getattr(ConsensusState, state_name, ConsensusState.COMMIT_ONE if current_val == 1 else ConsensusState.COMMIT_ZERO)

########## Manual implementation for Z3 simulator

def get_state_class():
    return ConsensusState

def get_all_transitions(num_nodes, num_rounds):
    """
    Transitions: [(current_state, [received_states], next_state), ...]
    """
    from itertools import product
    init_transitions = []
    transitions = []

    init_states = ConsensusState.get_initial_states() + [ConsensusState.get_lost_state()]
    all_initial_combs = list(product(init_states, repeat=num_nodes-1))
    current_states = [ConsensusState.LOCAL_ZERO, ConsensusState.LOCAL_ONE]
    for cur in current_states:
        for recv in all_initial_combs:
            next_state = transition(cur, list(recv), num_nodes)
            init_transitions.append((cur, recv, next_state))
    
    other_states = [s for s in ConsensusState if not s.is_initial]
    all_other_combs = list(product(other_states, repeat=num_nodes-1))
    current_states = other_states
    for cur in current_states:
        for recv in all_other_combs:
            if _filter_impossible_states([cur]+list(recv)):
                continue
            next_state = transition(cur, list(recv), num_nodes)
            transitions.append((cur, recv, next_state))
    
    return init_transitions, transitions

def get_state_name_mapping():
    return {
        "LocalZero": ConsensusState.LOCAL_ZERO.name,
        "LocalOne": ConsensusState.LOCAL_ONE.name,
        "Zero": ConsensusState.COMMIT_ZERO.name,
        "One": ConsensusState.COMMIT_ONE.name
    }

def get_transition_schema():
    return 0

def _filter_impossible_states(all_node_states):
    """
    Optional: Filter out impossible state combinations to reduce the search space (50x).
    """
    s_round = [False, False, False]
    reach_final = False
    for s in all_node_states:
        if s in [ConsensusState.S1_0, ConsensusState.S1_1]:
            s_round[0] = True
        elif s in [ConsensusState.S2_0, ConsensusState.S2_1]:
            s_round[1] = True
        elif s in [ConsensusState.S3_0, ConsensusState.S3_1]:
            s_round[2] = True
        elif s.is_final:
            reach_final = True
    if sum(s_round) > 1:
        return True
    if reach_final and sum(s_round) > 0:
        return True
    return False

########## Manual implementation for test cases

def fix_states():
    fixation = [
        (
            [
                ConsensusState.LOCAL_ONE,
                ConsensusState.LOCAL_ZERO,
                ConsensusState.LOCAL_ZERO,
                ConsensusState.LOCAL_ZERO,
                ConsensusState.LOCAL_ZERO,
                ConsensusState.LOCAL_ZERO,
            ], 0
        ),
        (
            [
                ConsensusState.CRASHED,
                ConsensusState.CRASHED,
                ConsensusState.CRASHED,
                ConsensusState.S3_1,
                ConsensusState.S3_0,
                ConsensusState.S3_0,
            ], 3
        )
    ]
    return fixation

def make_test_case():
    """
    6 nodes, 5 rounds (4 crashed)
    there's a gap between S3 and round 5
    """
    initial_states = [
        ConsensusState.LOCAL_ONE,
        ConsensusState.LOCAL_ZERO,
        ConsensusState.LOCAL_ZERO,
        ConsensusState.LOCAL_ZERO,
        ConsensusState.LOCAL_ZERO,
        ConsensusState.LOCAL_ZERO,
    ]
    round_msgs = [
        [
            (0, 1, 2, 3, 4, 5), # crashed
            (0, 1, 2, 3, 4, 5), # s1_1
            (1, 2, 3, 4, 5),
            (1, 2, 3, 4, 5),
            (1, 2, 3, 4, 5),
            (1, 2, 3, 4, 5),
        ], # r0
        [
            (),
            (1, 2, 3, 4, 5), # crashed
            (1, 2, 3, 4, 5), # s2_1
            (2, 3, 4, 5),
            (2, 3, 4, 5),
            (2, 3, 4, 5),
        ], # r1
        [
            (),
            (),
            (2, 3, 4, 5), # crashed
            (2, 3, 4, 5), # s3_1
            (3, 4, 5),
            (3, 4, 5),
        ], # r2
        [
            (),
            (),
            (),
            (3, 4, 5), # crashed
            (3, 4, 5), # commit_one
            (4, 5), # commit_zero
        ], # r3
        [
            (),
            (),
            (),
            (),
            (4, 5),
            (4, 5),
        ], # r4
    ]
    nodes = 6
    # init
    minism = MiniSM(num_state_machines=nodes)
    minism.initialize(protocol="ConsensusProtocol", initial_states=initial_states)
    logger.info(f"Initial States: {list(map(str, initial_states))}")
    # run the test case
    for r in range(len(round_msgs)):
        round_msg = round_msgs[r]
        last_round = (r == len(round_msgs) - 1)
        msgs = minism.get_msgs_list(round_msg, last_round=last_round)
        next_states = []
        for msg in msgs:
            cur_state, recvd_states = msg
            next_state = cur_state
            if len(recvd_states) != 0:
                next_state = transition(cur_state, recvd_states, total_nodes=nodes)
            next_states.append(next_state)
        minism.to_next(next_states)
        logger.info(f"Round {r}: {list(map(str, minism.get_global_state()))}")
