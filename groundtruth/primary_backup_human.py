from protocol.primary_backup import State

class PrimaryBackupHuman:
    """
    A human-designed protocol for the atomic commit scenario.
    """
    
    @classmethod
    def check_rounds(cls, rounds: int):
        """
        Checks if the number of rounds is valid for the AtomicCommit human-designed protocol.
        """
        assert rounds >= 1, "PrimaryBackupHuman protocol requires at least one rounds."
    
    @classmethod
    def get_actions(cls, messages_list: list[list[State]], is_last_round: int) -> list[State]:
        return [cls.get_action(messages, is_last_round) for messages in messages_list]

    @classmethod
    def get_action(cls, messages: list[State], is_last_round: int) -> State:
        """
        Given the current state and received messages, determines the next action
        according to the atomic commit human-designed protocol.

        is_last_round: 0 if not the last round, 1 if it is the last round.
        """

        # Pre-checks
        assert messages is not None and len(messages) > 0, "Messages cannot be None."
        if any(msg.is_final for msg in messages):
            raise ValueError("Should not have final states in the middle of the protocol.")

        should_one = any(msg is State.LocalOne or msg is State.DoNothing_One for msg in messages)
        if is_last_round:
            return State.One if should_one else State.Zero
        return State.DoNothing_One if should_one else State.DoNothing_Zero
