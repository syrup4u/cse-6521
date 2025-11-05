from protocol.state import AbstractState

class StateMachineManager:
    """ Manages all state machines as the global state. """

    def __init__(self, num_state_machines: int):
        self.state_machines: list[StateMachine] = [StateMachine() for _ in range(num_state_machines)]

    def initialize(self, protocol: str, initial_states: list[AbstractState]):
        """ Initializes all state machines with the given protocol and initial states. """

        assert len(initial_states) == len(self.state_machines), "Number of initial states must match number of state machines."

        for sm, init_state in zip(self.state_machines, initial_states):
            sm.initialize(protocol, init_state)

    def get_global_state(self) -> list[AbstractState]:
        """ Returns the global state of all state machines. """
        return [sm.get_final_state() for sm in self.state_machines]

    @staticmethod
    def apply_mask(states: list[AbstractState], mask: list[int]) -> list[AbstractState]:
        """
        Applies the given mask to the list of states, returning a new list with
        states included or replaced by lost_state based on the mask.
        """
        lost_state = AbstractState.get_lost_state()
        return [states[i] if i in mask else lost_state for i in range(len(states))]


class StateMachine:
    """
    State machine is just a simple recorder of states and messages.
    It does not have any logic of the protocol itself.
    The transition state is done externally, either by AI or human designed protocol.
    """

    def __init__(self):
        self.history_state: list[AbstractState] = None
        self.history_message: list[list[AbstractState]] = None
        self.protocol: str = None
        self.crashed: bool = False

    def initialize(self, protocol: str, initial_state: AbstractState):
        """ Initializes the state machine with the given protocol and initial state. """

        assert initial_state is not None and initial_state.is_initial, "Initial state must be a valid initial state."

        self.history_state = [initial_state]
        self.history_message = []
        self.protocol = protocol
        self.crashed = False

    def transition(self, next_state: AbstractState, messages: list[AbstractState]):
        """ Transitions the state machine to the next state and records the messages. """

        assert self.history_state is not None and self.history_message is not None, "State machine not initialized properly."

        self.history_state.append(next_state)
        self.history_message.append(messages)

    def get_initial_state(self) -> AbstractState:
        """ Returns the initial state of the state machine. """

        assert self.history_state is not None and len(self.history_state) > 0, "State machine not initialized properly."

        return self.history_state[0]
    
    def get_final_state(self) -> AbstractState:
        """ Returns the final state of the state machine. """

        assert self.history_state is not None and len(self.history_state) > 0, "State machine not initialized properly."

        return self.history_state[-1]
    
    def set_crashed(self):
        """ Marks the state machine as crashed. """
        self.crashed = True

    def get_current_round(self) -> int:
        """ Returns the current round index of the state machine. """

        assert self.history_state is not None, "State machine not initialized properly."

        return len(self.history_state)
