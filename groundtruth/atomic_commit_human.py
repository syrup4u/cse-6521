from protocol.atomic_commit import State

class AtomicCommitHuman:
    """
    A human-designed protocol for the atomic commit scenario.
    """
    
    @classmethod
    def check_rounds(cls, rounds: int):
        """
        Checks if the number of rounds is valid for the AtomicCommit human-designed protocol.
        """
        assert rounds >= 1, "AtomicCommitHuman protocol requires at least one rounds."
    
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

        first_round = any(msg.is_initial for msg in messages)
        # implement 1: any local abort should lead to global abort, any intermediate zero should abort (passive)
        # should_abort = any(msg is State.LocalAbort or msg is State.DoNothing_Zero for msg in messages)
        # implement 2: any local abort should lead to global abort, any intermediate one should commit (active)
        should_abort = any(msg is State.LocalAbort for msg in messages) \
            or (not first_round and not any(msg is State.DoNothing_One for msg in messages))

        if not should_abort and first_round:
            should_abort = any(msg is State.Lost for msg in messages)
        if should_abort:
            if is_last_round:
                return State.Abort
            else:
                return State.DoNothing_Zero
        else:
            if is_last_round:
                return State.Commit
            else:
                return State.DoNothing_One