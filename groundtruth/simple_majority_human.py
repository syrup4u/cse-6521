from protocol.simple_majority import State
from simulator.state_machine import StateMachine

class SimpleMajorityHuman:
    """
    A human-designed protocol for the simple majority voting scenario.
    """
    
    @classmethod
    def check_rounds(cls, rounds: int) -> bool:
        """
        Checks if the number of rounds is valid for the simple majority human-designed protocol.
        This protocol only supports one round.
        """
        return rounds == 1

    @classmethod
    def get_action(cls, sm: StateMachine, messages: list[State]) -> State:
        """
        Given the current state and received messages, determines the next action
        according to the simple majority human-designed protocol.
        """

        # Pre-checks
        if sm.crashed:
            return State.FAIL  # Crashed nodes cannot participate
        assert messages is not None, "Messages cannot be None."
        if any(msg in [State.FAIL, State.PASS] for msg in messages):
            raise ValueError("Should not have more than one round in SimpleMajorityHuman protocol.")

        # logic
        vote_yes = sum(1 for msg in messages if msg is State.YES)
        if vote_yes > len(messages) / 2:
            return State.PASS

        return State.FAIL
