from .state import AbstractState
from simulator.state_machine import StateMachine
import config

import logging

PROTOCOL_NAME = config.SUPPORT_PROTOCOLS[2]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class State(AbstractState):
    """
    Enumeration representing the possible states in an atomic commit protocol.
    """

    Zero = 0
    One = 1
    DoNothing_Zero = 2
    DoNothing_One = 3
    # below are not reachable during transition
    Lost = 4 # to represent crashed node
    LocalZero = 5
    LocalOne = 6

    @property
    def meaning(self) -> str:
        if self is State.Abort:
            return "Final Zero"
        if self is State.Commit:
            return "Final One"
        if self is State.DoNothing_Zero:
            return "Intermediate state: Likely Zero"
        if self is State.DoNothing_One:
            return "Intermediate state: Likely One"
        if self is State.Lost:
            return "Lost"
        if self is State.LocalAbort:
            return "Local Zero"
        if self is State.LocalCommit:
            return "Local One"
        return "Unknown state"
    
    @property
    def is_initial(self) -> bool:
        return self in {State.LocalZero, State.LocalOne}
    
    @property
    def is_final(self) -> bool:
        return self in {State.Zero, State.One}

    @classmethod
    def get_lost_state(cls) -> 'State':
        return State.Lost
    
    @classmethod
    def get_initial_states(cls) -> list['State']:
        return [State.LocalZero, State.LocalOne]


class PrimaryBackupProtocol:
    """
    Atomic Commit Protocol
    """

    REWARD = {
        "good": 1,
        "bad": -4
    }

    @classmethod
    def get_reward(cls, global_state: list[StateMachine]) -> int:
        """
        Computes the reward based on the global states of all state machines
        according to the rules of the atomic commit protocol.
        """

        assert global_state is not None and len(global_state) > 0, "Global state must contain at least one state machine."
        assert all(sm.protocol == PROTOCOL_NAME for sm in global_state), "All state machines must use the atomic commit protocol."

        final_states = [sm.get_final_state() for sm in global_state if sm.get_final_state() is not State.Lost]

        # Rule 1: all uncrashed nodes must reach a final decision
        if any(not s.is_final for s in final_states):
            logger.debug("Not all uncrashed nodes reached a final decision.")
            return cls.REWARD["bad"]
        
        # Rule 2: only make decision at the end
        for i, sm in enumerate(global_state):
            all_intermediate_state = sm.history_state[:-1]
            if State.Zero in all_intermediate_state or State.One in all_intermediate_state:
                logger.debug(f"Node {i} made a final decision before the end.")
                logger.debug(f"Node {i} history states: {all_intermediate_state}")
                return cls.REWARD["bad"]
        
        # Rule 3: No decision contradicts any other decision
        if State.Zero in final_states and State.One in final_states:
            logger.debug("Conflicting final decisions detected: both Abort and Commit present.")
            return cls.REWARD["bad"]

        # Rule 4: Final decision must have corresponding initial states
        common_final_state = final_states[0]
        all_initial = sum(sm.get_initial_state().value for sm in global_state)
        all_initial_zero = all_initial == State.LocalZero.value * len(global_state)
        all_initial_one = all_initial == State.LocalOne.value * len(global_state)
        if all_initial_zero and common_final_state == State.One:
            logger.debug("All initial states are LocalZero but final decision is One.")
            return cls.REWARD["bad"]
        if all_initial_one and common_final_state == State.Zero:
            logger.debug("All initial states are LocalOne but final decision is Zero.")
            return cls.REWARD["bad"]

        # Acceptable cases
        return cls.REWARD["good"]