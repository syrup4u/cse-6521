from protocol import AbstractState

from typing import List, Set

class ConsensusState(AbstractState):
    """
    Crash-tolerant consensus protocol with 2-round synchronous execution.
    
    Protocol guarantees:
    - Agreement: All live nodes commit to the same value
    - Validity: Commit value must have existed in initial configuration
    - Termination: All live nodes reach final state in 2 rounds
    
    State progression:
    Round 1: INITIAL_* → KNOWLEDGE_* (collect observed initial values)
    Round 2: KNOWLEDGE_* → COMMIT_* (compute intersection of knowledge sets)
    """
    # Unique enum values using distinct integers as primary identifiers
    INITIAL_ZERO = (0, True, False)
    INITIAL_ONE = (1, True, False)
    KNOWLEDGE_ZERO_ONLY = (2, False, False)
    KNOWLEDGE_ONE_ONLY = (3, False, False)
    KNOWLEDGE_BOTH = (4, False, False)
    COMMIT_ZERO = (5, False, True)
    COMMIT_ONE = (6, False, True)
    LOST = (7, False, True)
    
    def __init__(self, _id: int, is_initial: bool, is_final: bool):
        # _id ensures unique enum values; properties stored as attributes
        self._is_initial = is_initial
        self._is_final = is_final
    
    @property
    def is_initial(self) -> bool:
        return self._is_initial
    
    @property
    def is_final(self) -> bool:
        return self._is_final
    
    @classmethod
    def get_lost_state(cls) -> 'ConsensusState':
        return ConsensusState.LOST
    
    @classmethod
    def get_initial_states(cls) -> List['ConsensusState']:
        return [ConsensusState.INITIAL_ZERO, ConsensusState.INITIAL_ONE]
    
    @classmethod
    def get_final_states(cls) -> List['ConsensusState']:
        return [ConsensusState.COMMIT_ZERO, ConsensusState.COMMIT_ONE]


def transition(current_state: ConsensusState, received_messages: List[ConsensusState], num_nodes) -> ConsensusState:
    """
    Two-round consensus protocol transition function.
    
    Round 1 behavior:
      - Node broadcasts its initial value (0/1)
      - Collects all initial values seen (including own)
      - Transitions to knowledge state representing observed values
    
    Round 2 behavior:
      - Node broadcasts its knowledge set
      - Computes intersection of all received knowledge sets (including own)
      - Decides based on intersection:
        * {0} → commit_zero
        * {1} → commit_one
        * ∅ or {0,1} → commit_zero (valid tie-break since both values existed initially)
    
    Crash handling:
      - Crashed nodes stop broadcasting (handled by simulator)
      - Live nodes proceed with whatever messages they received
      - Intersection operation naturally handles missing messages
    """
    # Terminal states remain unchanged
    if current_state.is_final:
        return current_state
    
    # ===== ROUND 1: Build knowledge set from initial values =====
    if current_state == ConsensusState.INITIAL_ZERO or current_state == ConsensusState.INITIAL_ONE:
        # Start with own value
        knowledge: Set[int] = {0} if current_state == ConsensusState.INITIAL_ZERO else {1}
        
        # Add values from received initial states
        for msg in received_messages:
            if msg == ConsensusState.INITIAL_ZERO:
                knowledge.add(0)
            elif msg == ConsensusState.INITIAL_ONE:
                knowledge.add(1)
            # Ignore non-initial messages (shouldn't occur in correct round 1)
        
        # Transition to appropriate knowledge state
        if knowledge == {0}:
            return ConsensusState.KNOWLEDGE_ZERO_ONLY
        elif knowledge == {1}:
            return ConsensusState.KNOWLEDGE_ONE_ONLY
        else:  # knowledge == {0, 1}
            return ConsensusState.KNOWLEDGE_BOTH
    
    # ===== ROUND 2: Compute intersection of knowledge sets =====
    elif current_state in (
        ConsensusState.KNOWLEDGE_ZERO_ONLY,
        ConsensusState.KNOWLEDGE_ONE_ONLY,
        ConsensusState.KNOWLEDGE_BOTH
    ):
        # Determine own knowledge set
        if current_state == ConsensusState.KNOWLEDGE_ZERO_ONLY:
            own_set = {0}
        elif current_state == ConsensusState.KNOWLEDGE_ONE_ONLY:
            own_set = {1}
        else:  # KNOWLEDGE_BOTH
            own_set = {0, 1}
        
        # Initialize intersection with own knowledge
        intersection = own_set.copy()
        
        # Intersect with knowledge sets from received messages
        for msg in received_messages:
            if msg == ConsensusState.KNOWLEDGE_ZERO_ONLY:
                intersection &= {0}
            elif msg == ConsensusState.KNOWLEDGE_ONE_ONLY:
                intersection &= {1}
            elif msg == ConsensusState.KNOWLEDGE_BOTH:
                intersection &= {0, 1}
            # Ignore invalid messages (shouldn't occur in correct round 2)
        
        # Decision logic with validity preservation
        if intersection == {1}:
            return ConsensusState.COMMIT_ONE
        else:  # intersection == {0} OR {0,1} OR ∅
            # Validity guarantee: 
            # - {0} → 0 existed initially ✓
            # - {0,1} → both existed initially ✓
            # - ∅ → only possible when both values existed but partitioned (e.g., node saw only 0s, another only 1s)
            #        Since both existed initially, committing 0 preserves validity
            return ConsensusState.COMMIT_ZERO
    
    # Should never reach here in valid executions
    return current_state

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
    current_states = [ConsensusState.INITIAL_ZERO, ConsensusState.INITIAL_ONE]
    for cur in current_states:
        for recv in all_initial_combs:
            next_state = transition(cur, list(recv), num_nodes)
            init_transitions.append((cur, recv, next_state))
    
    other_states = [s for s in ConsensusState if not s.is_initial]
    print(other_states)
    print(ConsensusState.COMMIT_ONE)
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
        "LocalZero": ConsensusState.INITIAL_ZERO.name,
        "LocalOne": ConsensusState.INITIAL_ONE.name,
        "Zero": ConsensusState.COMMIT_ZERO.name,
        "One": ConsensusState.COMMIT_ONE.name
    }

def get_transition_schema():
    return 0

def _filter_impossible_states(all_node_states):
    return False
