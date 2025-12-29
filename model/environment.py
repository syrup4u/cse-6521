from protocol.state import AbstractState

import torch

class Environment:
    """
    Environment wrapper for the StateMachineManager to interact with RL agents.
    Transform basic types to tensor and vice versa.
    """

    def __init__(self, state_class: type[AbstractState], offset: int = 0, device='cpu'):
        """
        offset: action offset for the state representation.
            E.g., 0/1 is the initial state, 2/3/4/5 is the possible action space,
            then offset = 2 to align the state representation with action indices.
        """
        self.state_class = state_class
        self.state_offset = offset if offset > 0 else 0
        self.device = device

    def get_state_all(self, msgs_list: list[list[AbstractState]], round: int) -> torch.Tensor:
        """
        msgs_list: shape N x N, where N is the number of state machines (nodes).

        Returns the current global state as a tensor.
        - Tensor shape: (num_state_machines, num_state_machines+1),
        where the first dimension can be seen as the batch size (N),
        the second dimension represents the sequence set (N+1), including round number (or is_last_round flag)
        """
        msg_tensor = torch.tensor(
            [[msg.value for msg in msgs] for msgs in msgs_list],
            dtype=torch.long,
            device=self.device
        ) # (N, N)
        round_tensor = torch.full_like(msg_tensor[:, :1], fill_value=len(self.state_class)+round) # (N, 1)
        state_tensor = torch.cat([round_tensor, msg_tensor], dim=-1) # (N, N+1)
        return state_tensor

    def step_all(self, actions: torch.Tensor) -> list[AbstractState]:
        """
        actions: (num_state_machines,) tensor
        """
        next_states = []
        actions = actions + self.state_offset
        for action in actions:
            next_state = self.state_class(action.item())
            next_states.append(next_state)
        return next_states


class L2Transformer:
    """
    Transform basic types to tensor and vice versa for L2.
    Map L2's output back to true final states.
    """

    def __init__(self, state_class: type[AbstractState], offset: int = 0, device='cpu'):
        self.state_class = state_class
        self.state_offset = offset
        self.device = device

    def get_state_all(self, msgs_list: list[list[AbstractState]]) -> torch.Tensor:
        """
        No round encoding version of get_state_all.

        msgs_list: shape len(subsets) * subset_size x subset_size
        """
        msg_tensor = torch.tensor(
            [[msg.value for msg in msgs] for msgs in msgs_list],
            dtype=torch.long,
            device=self.device
        ) # (N, N)
        return msg_tensor

    def step_all(self, actions: torch.Tensor) -> list[AbstractState]:
        next_states = []
        actions = actions + self.state_offset
        for action in actions:
            next_state = self.state_class(action.item())
            next_states.append(next_state)
        return next_states
