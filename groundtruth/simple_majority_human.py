from protocol.simple_majority import State

class SimpleMajorityHuman:
    """
    A human-designed protocol for the simple majority voting scenario.
    """
    
    @classmethod
    def check_rounds(cls, rounds: int):
        """
        Checks if the number of rounds is valid for the simple majority human-designed protocol.
        This protocol only supports one round.
        """
        assert rounds == 1, "SimpleMajorityHuman protocol only supports one round."
    
    @classmethod
    def get_actions(cls, messages_list: list[list[State]], round_idx: int) -> list[State]:
        return [cls.get_action(messages, round_idx) for messages in messages_list]

    @classmethod
    def get_action(cls, messages: list[State], round_idx: int) -> State:
        """
        Given the current state and received messages, determines the next action
        according to the simple majority human-designed protocol.
        """

        # Pre-checks
        assert messages is not None, "Messages cannot be None."
        if any(msg in [State.FAIL, State.PASS] for msg in messages):
            raise ValueError("Should not have more than one round in SimpleMajorityHuman protocol.")

        # logic
        vote_yes = sum(1 for msg in messages if msg is State.YES)
        if vote_yes > len(messages) / 2:
            return State.PASS

        return State.FAIL
