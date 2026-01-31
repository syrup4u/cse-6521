from simulator.state_machine import StateMachineManager

class MiniSM:
    """ A minimal state machine for testing purposes. """

    def __init__(self, num_state_machines: int):
        self.manager = StateMachineManager(num_state_machines)

    def get_global_state(self) -> list:
        return self.manager.get_global_state()

    def initialize(self, protocol: str, initial_states: list):
        self.manager.initialize(protocol, initial_states)

    def get_msgs_list(self, round_msg: list, last_round=False):
        global_states = self.manager.get_global_state()
        msgs_list = []
        alive_nodes = list(range(len(global_states)))
        for i, msg in enumerate(round_msg):
            msg_other_states = []
            for j in range(len(global_states)):
                if j != i:
                    if j in msg:
                        msg_other_states.append(global_states[j])
                    else:
                        msg_other_states.append(global_states[j].get_lost_state())
            msgs_list.append((global_states[i], msg_other_states))
            if len(msg) > 0 and len(msg) < len(alive_nodes):
                # some nodes are crashed
                alive_nodes = list(msg)
        # mark crashed nodes
        if not last_round:
            for i in range(len(global_states)):
                if i not in alive_nodes:
                    self.manager.state_machines[i].set_crashed()
        return msgs_list

    def to_next(self, next_states: list):
        for sm, next_state in zip(self.manager.state_machines, next_states):
            if not sm.crashed:
                sm.transition(next_state, [])
            else:
                sm.transition(next_state.get_lost_state(), [])
