from protocol.simple_majority import SimpleMajorityProtocol
from simulator.state_machine import StateMachine

# TODO: protocol can be an instance instead of a string
def verify(protocol: str, global_state: list[StateMachine]) -> bool:
    if protocol == "simple_majority":
        reward = SimpleMajorityProtocol.get_reward(global_state)
        if reward >= 0:
            return True
        return False
    
    raise NotImplementedError(f"Verification for protocol {protocol} is not implemented.")
