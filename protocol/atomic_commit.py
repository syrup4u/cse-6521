from .state import AbstractState
from simulator.state_machine import StateMachine
import config

import logging

PROTOCOL_NAME = config.SUPPORT_PROTOCOLS[1]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class State(AbstractState):
    """
    Enumeration representing the possible states in an atomic commit protocol.
    """

    Abort = 0
    Commit = 1
    DoNothing_Zero = 2
    DoNothing_One = 3
    # below are not reachable during transition
    Lost = 4 # to represent crashed node
    LocalAbort = 5
    LocalCommit = 6

    @property
    def meaning(self) -> str:
        if self is State.Abort:
            return "Final Abort"
        if self is State.Commit:
            return "Final Commit"
        if self is State.DoNothing_Zero:
            return "May Abort"
        if self is State.DoNothing_One:
            return "Likely Commit"
        if self is State.Lost:
            return "Lost"
        if self is State.LocalAbort:
            return "Local Abort"
        if self is State.LocalCommit:
            return "Local Commit"
        return "Unknown state"
    
    @property
    def is_initial(self) -> bool:
        return self in {State.LocalAbort, State.LocalCommit}
    
    @property
    def is_final(self) -> bool:
        return self in {State.Abort, State.Commit}

    @classmethod
    def get_lost_state(cls) -> 'State':
        return State.Lost
    
    @classmethod
    def get_initial_states(cls) -> list['State']:
        return [State.LocalAbort, State.LocalCommit]

    @classmethod
    def get_final_states(cls) -> list['State']:
        return [State.Abort, State.Commit]


class AtomicCommitProtocol:
    """
    Atomic Commit Protocol
    """

    REWARD = {
        "bonus": 2,
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
            if State.Abort in all_intermediate_state or State.Commit in all_intermediate_state:
                logger.debug(f"Node {i} made a final decision before the end.")
                logger.debug(f"Node {i} history states: {all_intermediate_state}")
                return cls.REWARD["bad"]
        
        # Rule 3: No decision contradicts any other decision
        if State.Abort in final_states and State.Commit in final_states:
            logger.debug("Conflicting final decisions detected: both Abort and Commit present.")
            return cls.REWARD["bad"]

        # Rule 4: If all initial states are LocalCommit and there is no crash, then all final states must be Commit
        common_final_state = final_states[0]
        all_initial_commit = all(sm.get_initial_state() is State.LocalCommit for sm in global_state)
        no_crash = all(not sm.crashed for sm in global_state)
        if all_initial_commit and no_crash:
            # if any(sm.get_final_state() is not State.Commit for sm in global_state):
            if common_final_state is not State.Commit:
                logger.debug("All initial states are LocalCommit with no crashes, but not all final states are Commit.")
                return cls.REWARD["bad"]
            
        # Rule 5: If any initial state is LocalAbort, then all final states must be Abort
        if any(sm.get_initial_state() is State.LocalAbort for sm in global_state):
            # if any(sm.get_final_state() is not State.Abort for sm in reach_final_nodes):
            if common_final_state is not State.Abort:
                logger.debug("At least one initial state is LocalAbort, but not all valid final states are Abort.")
                return cls.REWARD["bad"]

        # Acceptable cases
        """
        Encourage Commit if possible for liveness, but may lead to aggressive behavior 
        which is bad for generalization.
        """
        # if common_final_state is State.Commit:
        #     logger.info("Bonus for:")
        #     for sm in global_state:
        #         logger.info(sm.history_state)
        #     return cls.REWARD["bonus"]

        return cls.REWARD["good"]