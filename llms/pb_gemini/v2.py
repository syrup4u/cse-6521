from dataclasses import dataclass

class ConsensusState:
    def __init__(self, value: int, round_num: int, state_type: str = "PROCEEDING"):
        self.val = value
        self.round = round_num
        self.state_type = state_type # "INITIAL", "PROCEEDING", "FINAL", "CRASHED"

    @property
    def is_initial(self) -> bool: return self.state_type == "INITIAL"
    
    @property
    def is_final(self) -> bool: return self.state_type == "FINAL"
    
    @property
    def is_crashed(self) -> bool: return self.state_type == "CRASHED"

    # For compatibility with your previous comparison logic
    def __eq__(self, other):
        if not isinstance(other, ConsensusState): return False
        return (self.val, self.round, self.state_type) == (other.val, other.round, other.state_type)

def transition(current: ConsensusState, received: list[ConsensusState], N: int) -> ConsensusState:
    if current.is_final or current.is_crashed:
        return current

    # 1. Update value: If current or any received message is 1, adopt 1
    # This is the "Flood" part of the protocol
    new_val = current.val
    if new_val == 0:
        for msg in received:
            if not msg.is_crashed and msg.val == 1:
                new_val = 1
                break
    
    next_round = current.round + 1

    # 2. Decision logic: Must wait for N rounds to guarantee agreement
    if next_round >= N:
        return ConsensusState(new_val, 99, "FINAL")
    
    return ConsensusState(new_val, next_round, "PROCEEDING")