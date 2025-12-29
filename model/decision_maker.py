from .base import *
import config

import torch
import torch.nn as nn
from torch import optim
import logging

logger = logging.getLogger(__name__)

class DecisionMaker(nn.Module):
    """
    Second layer (learnable): make final decision based on the final states collected from all subsets.

    Supports two types of models: [SetTransformer, PoolingHuman].
    TODO: still has some problems to be solved.
    """
    def __init__(self, dm: nn.Module):
        super(DecisionMaker, self).__init__()
        self.dm = dm
        self.optimizer = None
        self.input_encoder = nn.Identity

    def make_decision(self, final_states: torch.Tensor) -> torch.Tensor:
        """
        final_states: (N, sequence of final states)
        """
        logits = self.dm(self.input_encoder(final_states)) # (B, n_actions)
        final_decision = torch.argmax(logits, dim=-1)
        return final_decision  # (N,)

    def update(self):
        pass

    def save_model(self, path: str):
        torch.save(self.dm.state_dict(), path)

    def load_model(self, path: str):
        self.dm.load_state_dict(torch.load(path))


class PoolingHuman(nn.Module):
    """
    A simple human decision maker that uses pooling over the collected final states. (Fixed, only for evaluation)

    Supports two types of pooling: ['any', 'majority'].
    """
    def __init__(self, num_states: int, pooling_type='any', others: list = []):
        super(PoolingHuman, self).__init__()
        self.num_states = num_states
        if pooling_type == 'any':
            self.specified_state = others[0] if others else 0
            self.pooling = self._any_pooling
        elif pooling_type == 'majority':
            self.pooling = self._majority_pooling
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, sequence of final states)

        Sequence should have the values in [0, num_states-1].
        """
        return self.pooling(x)

    def _any_pooling(self, x: torch.Tensor) -> torch.Tensor:
        # if any specified state appears, return that state
        mask = (x == self.specified_state) # (N, sequence length)
        any_found = mask.any(dim=-1).unsqueeze(-1) # (N, 1)
        result = torch.where(
            any_found,
            torch.scatter(
                torch.zeros(x.size(0), self.num_states, device=x.device, dtype=x.dtype),
                dim=1,
                index=torch.full((x.size(0), 1), self.specified_state, device=x.device, dtype=torch.long),
                src=torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
            ),
            torch.scatter(
                torch.zeros(x.size(0), self.num_states, device=x.device, dtype=x.dtype),
                dim=1,
                index=x[:, 0:1],
                src=torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
            ) # default to the first state if not found (binary required)
        )
        return result

    def _majority_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, sequence of final states)
        returns: (N, 1) tensor
        """
        counts = torch.zeros(x.size(0), self.num_states, device=x.device, dtype=x.dtype)
        counts.scatter_add_(1, x, torch.ones_like(x))
        return counts


def build_set_transformer_model(
        dim_output: int,
        device: str = 'cpu'
    ) -> DecisionMaker:
    model = SetTransformer(
        dim_input=config.SET_TRANSFORMER_CONFIG["dim_input"],
        dim_output=dim_output,
        num_inds=config.SET_TRANSFORMER_CONFIG["num_inds"],
        dim_hidden=config.SET_TRANSFORMER_CONFIG["dim_hidden"],
        num_heads=config.SET_TRANSFORMER_CONFIG["num_heads"],
        num_outputs=config.SET_TRANSFORMER_CONFIG["num_outputs"],
        num_states=dim_output,
        num_rounds=0,
        encode_round_number=True
    ).to(device)
    dm = DecisionMaker(model)
    dm.optimizer = optim.Adam(model.parameters(), lr=config.DQN_CONFIG["learning_rate"]) # TODO: modify lr
    dm.input_encoder = model.tok_emb
    return dm

def build_human_model(
        num_states: int,
        pooling_type: str = 'any',
        state_offset: int = 0,
        specified_state: int = 0,
        device: str = 'cpu'
    ) -> DecisionMaker:
    model = PoolingHuman(
        num_states=num_states,
        pooling_type=pooling_type,
        others=[specified_state-state_offset]
    ).to(device)
    dm = DecisionMaker(model)
    dm.input_encoder = lambda x: x - state_offset
    return dm
