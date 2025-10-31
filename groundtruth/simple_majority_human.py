from protocol.simple_majority import State

class SimpleMajorityHuman:
    """
    A human-designed protocol for the simple majority voting scenario.
    """

    @classmethod
    def get_action(cls, cur_state: State, messages: list[State]) -> State:
        """
        Given the current state and received messages, determines the next action
        according to the simple majority human-designed protocol.
        """

        assert messages is not None, "Messages cannot be None."

        vote_yes = sum(1 for msg in messages if msg is State.YES)
        if vote_yes > len(messages) / 2:
            return State.PASS
        return State.FAIL
