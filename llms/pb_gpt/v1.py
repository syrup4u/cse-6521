from protocol import AbstractState

from typing import List, FrozenSet

# --------------------------
# Protocol State Definition
# --------------------------
class State(AbstractState):
    """
    Enum members carry:
      - seen: frozenset of strings, elements are '0' or '1' representing which initial values
      - is_initial: bool
      - is_final: bool
      - label: string (for debugging/clarity)
    """
    LOCAL_ZERO   = (frozenset({'0'}), True,  False, 'local_zero')
    LOCAL_ONE    = (frozenset({'1'}), True,  False, 'local_one')
    SEEN_0       = (frozenset({'0'}), False, False, 'seen_0')
    SEEN_1       = (frozenset({'1'}), False, False, 'seen_1')
    SEEN_BOTH    = (frozenset({'0','1'}), False, False, 'seen_both')
    COMMIT_ZERO  = (frozenset({'0'}), False, True,  'commit_zero')
    COMMIT_ONE   = (frozenset({'1'}), False, True,  'commit_one')
    LOST         = (frozenset(),        False, False, 'lost')

    def __init__(self, seen: FrozenSet[str], is_initial: bool, is_final: bool, label: str):
        self._seen = seen
        self._is_initial = is_initial
        self._is_final = is_final
        self._label = label

    # AbstractState required properties / classmethods
    @property
    def is_initial(self) -> bool:
        return self._is_initial

    @property
    def is_final(self) -> bool:
        return self._is_final

    @classmethod
    def get_lost_state(cls) -> 'State':
        return cls.LOST

    @classmethod
    def get_initial_states(cls) -> list['State']:
        return [cls.LOCAL_ZERO, cls.LOCAL_ONE]

    @classmethod
    def get_final_states(cls) -> list['State']:
        return [cls.COMMIT_ZERO, cls.COMMIT_ONE]

    # convenience
    @property
    def seen(self) -> FrozenSet[str]:
        return self._seen

    def __repr__(self) -> str:
        return f"<State.{self._label}>"

# --------------------------
# Transition function
# --------------------------
def transition(current: State, received: List[State], num_nodes) -> State:
    """
    Compute next state given `current` state and list of `received` states from other nodes in this round.

    Rules:
      1. If any received state is a COMMIT_* then adopt that same commit immediately.
      2. Otherwise, compute union_seen = union of current.seen and all seen sets encoded in received states.
      3. If union_seen == current.seen -> local view stabilized: deterministically commit:
           - if union_seen == {'0'} -> COMMIT_ZERO
           - if union_seen == {'1'} -> COMMIT_ONE
           - if union_seen == {'0','1'} -> tie-break -> COMMIT_ZERO (deterministic)
      4. Otherwise (union grew/changed) -> return an intermediate SEEN_* state that encodes union_seen
         (these states cause the node to continue flooding the larger seen set next round).
      5. If weird empty-seen (shouldn't happen for a properly initialized node) -> LOST
    """
    # 1) Immediate adoption of an explicit commit seen in messages
    for msg in received:
        if msg is State.COMMIT_ZERO:
            return State.COMMIT_ZERO
        if msg is State.COMMIT_ONE:
            return State.COMMIT_ONE

    # 2) compute union of seen sets (including our own)
    union_seen = set(current.seen)
    for msg in received:
        # treat any state by its encoded 'seen' set, so intermediate / initial / commit all contribute
        if isinstance(msg, State):
            union_seen.update(msg.seen)

    # 3) If union_seen same as current -> stability -> commit deterministically
    if frozenset(union_seen) == current.seen:
        if union_seen == {'0'}:
            return State.COMMIT_ZERO
        if union_seen == {'1'}:
            return State.COMMIT_ONE
        if union_seen == {'0', '1'}:
            # deterministic tie-break when both initial values exist: pick zero
            return State.COMMIT_ZERO
        # empty => lost
        return State.LOST

    # 4) Otherwise return an intermediate state encoding the new union
    if union_seen == {'0'}:
        return State.SEEN_0
    if union_seen == {'1'}:
        return State.SEEN_1
    if union_seen == {'0', '1'}:
        return State.SEEN_BOTH

    # defensive fallback
    return State.LOST

########## Manual implementation for Z3 simulator

def get_state_class():
    return State

def get_all_transitions(num_nodes, num_rounds):
    """
    Transitions: [(current_state, [received_states], next_state), ...]
    """
    from itertools import product
    init_transitions = []
    transitions = []

    init_states = State.get_initial_states() + [State.get_lost_state()]
    all_initial_combs = list(product(init_states, repeat=num_nodes-1))
    current_states = [State.LOCAL_ZERO, State.LOCAL_ONE]
    for cur in current_states:
        for recv in all_initial_combs:
            next_state = transition(cur, list(recv), num_nodes)
            init_transitions.append((cur, recv, next_state))
    
    other_states = [s for s in State if not s.is_initial]
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
        "LocalZero": State.LOCAL_ZERO.name,
        "LocalOne": State.LOCAL_ONE.name,
        "Zero": State.COMMIT_ZERO.name,
        "One": State.COMMIT_ONE.name
    }

def get_transition_schema():
    return 0

def _filter_impossible_states(all_node_states):
    """
    Optional: Filter out impossible state combinations to reduce the search space (50x).
    """
    return False
