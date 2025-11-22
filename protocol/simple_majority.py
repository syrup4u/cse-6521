from .state import AbstractState
from simulator.state_machine import StateMachine
import config

PROTOCOL_NAME = config.SUPPORT_PROTOCOLS[0]
ROUND = 1

class State(AbstractState):
    """
    Enumeration representing the possible states in a simple majority voting protocol.
    """

    NO = 0
    YES = 1
    FAIL = 2
    PASS = 3

    @property
    def meaning(self) -> str:
        if self is State.NO:
            return "Local votes no"
        if self is State.YES:
            return "Local votes yes"
        if self is State.FAIL:
            return "Final decision fail"
        if self is State.PASS:
            return "Final decision pass"
        return "Unknown state"
    
    @property
    def is_initial(self) -> bool:
        return self in {State.NO, State.YES}
    
    @property
    def is_final(self) -> bool:
        return self in {State.FAIL, State.PASS}

    @classmethod
    def get_lost_state(cls) -> 'State':
        return State.NO
    
    @classmethod
    def get_initial_states(cls) -> list['State']:
        return [State.NO, State.YES]

class SimpleMajorityProtocol:
    """
    This is not a practical protocol but enough for demonstration purposes.
    """

    REWARD = {
        "good": 1,
        "bad": -1,
        "neutral": 0
    }

    @classmethod
    def get_reward(cls, global_state: list[StateMachine]) -> int:
        """
        Computes the reward based on the global states of all state machines
        according to the rules of the simple majority protocol.
        """

        assert global_state is not None and len(global_state) > 0, "Global state must contain at least one state machine."
        assert all(sm.protocol == PROTOCOL_NAME for sm in global_state), "All state machines must use the simple majority protocol."

        num_nodes = len(global_state)
        num_vote_yes_valid = sum(1 for sm in global_state if not sm.crashed and sm.get_initial_state() is State.YES)
        num_vote_no_valid = sum(1 for sm in global_state if not sm.crashed and sm.get_initial_state() is State.NO)

        # Rule 1: all uncrashed nodes must reach a final decision
        if any(not sm.get_final_state().is_final for sm in global_state if not sm.crashed):
            return cls.REWARD["bad"]
        
        # Rule 2: majority voting correctness (since there is only one round)
        if num_vote_yes_valid > num_nodes / 2:
            # Majority voted YES without crashing
            if any(sm.get_final_state() is State.FAIL for sm in global_state if not sm.crashed):
                return cls.REWARD["bad"]
            else:
                return cls.REWARD["good"]

        # Rule 3: majority voting correctness (since there is only one round)
        if num_vote_no_valid >= (num_nodes + 1) / 2:
            # Majority voted NO without crashing
            if any(sm.get_final_state() is State.PASS for sm in global_state if not sm.crashed):
                return cls.REWARD["bad"]
            else:
                return cls.REWARD["good"]
            
        # Ambiguous case: no clear majority, then the reward is neutral,
        # any final decision is possible and conflicts are allowed.
        return cls.REWARD["neutral"]
