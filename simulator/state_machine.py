from protocol.state import AbstractState

from typing import List

class StateMachine:
    """
    State machine is just a simple recorder of states and messages.
    It does not have any logic of the protocol itself.
    The transition state is done externally, either by AI or human designed protocol.
    """

    def __init__(self):
        self.history_state: List[AbstractState] = None
        self.history_message: List[List[AbstractState]] = None
        self.protocol: str = None
        self.crashed: bool = False

    def initialize(self, protocol: str, initial_state: AbstractState):
        """
        Initializes the state machine with the given protocol and initial state.
        """

        assert initial_state is not None and initial_state.is_initial, "Initial state must be a valid initial state."

        self.history_state = [initial_state]
        self.history_message = []
        self.protocol = protocol
        self.crashed = False

    def transition(self, next_state: AbstractState, messages: List[AbstractState]):
        """
        Transitions the state machine to the next state and records the messages.
        """

        assert self.history_state is not None and self.history_message is not None, "State machine not initialized properly."

        self.history_state.append(next_state)
        self.history_message.append(messages)

    def get_initial_state(self) -> AbstractState:
        """
        Returns the initial state of the state machine.
        """

        assert self.history_state is not None and len(self.history_state) > 0, "State machine not initialized properly."

        return self.history_state[0]
    
    def get_final_state(self) -> AbstractState:
        """
        Returns the final state of the state machine.
        """

        assert self.history_state is not None and len(self.history_state) > 0, "State machine not initialized properly."

        return self.history_state[-1]
    
    def set_crashed(self):
        """
        Marks the state machine as crashed.
        """

        self.crashed = True
