import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.set_transformer.modules import ISAB, PMA, SAB

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        - input_size: Input feature size
        - hidden_sizes: List of hidden layer sizes
        - output_size: Output feature size
        """

        super(MLP, self).__init__()
        self.input_size = input_size
        layers = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        layers.append(nn.Linear(in_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)  # shape: (B, output_size)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        One-hot encode the input sequence tensor, and take the pooling over the sequence dimension.
        - x: (B, N+1) tensor
        - returns: (B, input_size) tensor, where input_size = num_states + num_rounds

        The reason to include current round as part of the encoding is to help the model reach final decision at the last round. (for generalization purpose, use round number instead of just a binary flag indicating last round)
        """
        x_onehot = F.one_hot(x.long(), num_classes=self.input_size).float()  # (B, N+1, C)
        return x_onehot.mean(dim=1)  # (B, C)


class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input=3,
        dim_output=40,
        num_inds=4,
        dim_hidden=128,
        num_heads=4,
        num_outputs=1,
        num_states=4,
        num_rounds=1,
        ln=False,
    ):
        """
        - num_inds: Number of inducing points (learnable points to condense information from the set)
        - dim_hidden: Dimension of the hidden layers
        - num_heads: Number of attention heads (each head has dim_hidden / num_heads dimensions)
        - num_outputs: Number of query and output vectors in decoder
        - num_states: number of possible states in the protocol state machine
        - num_rounds: number of total rounds for round number embedding
        """
        super(SetTransformer, self).__init__()
        if num_inds >= dim_input: # no need to use inducing points
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            )
        else:
            self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        )

        self.tok_emb = nn.Embedding(num_states + num_rounds, dim_input) # B x (N+1) -> x D

        # Mark: do we need to tie the embedding weights with the output layer weights? No.
        # self.dec[-1].weight = self.tok_emb.weight
    
    def forward(self, X):
        return self.dec(self.enc(X)).squeeze(1) # (B, dim_output)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)

class PolicyConstraint:
    """
    Additional constraints on the policy network outputs.
    """
    
    @classmethod
    def get_action(cls, logits: torch.Tensor, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        - states: (B, N+1) tensor, where B is the batch size or number of state machines
        - logits: (B, num_actions) tensor
        """
        # states = states.reshape((states.shape[0], -1))
        # logits = logits.reshape((-1, logits.shape[-1]))

        # get unique states and their logits
        sorted_states, _ = torch.sort(states, dim=-1)
        _, inverse_indices, counts = torch.unique(sorted_states, dim=0, return_inverse=True, return_counts=True)

        num_unique = counts.shape[0]
        unique_logits = torch.zeros((num_unique, logits.shape[-1]), device=logits.device, dtype=logits.dtype)
        unique_logits.index_add_(0, inverse_indices, logits) # since logits are same for same unique states, we can sum them up
        unique_logits = unique_logits / counts.unsqueeze(-1)

        # now we can sample actions for unique states
        m = torch.distributions.Categorical(logits=unique_logits)
        unique_actions = m.sample()
        unique_log_probs = m.log_prob(unique_actions)

        # map back to original states' actions and log_probs
        action = unique_actions[inverse_indices]
        log_probs = unique_log_probs[inverse_indices]

        return action, log_probs

    @classmethod
    def force_same_action(cls, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        - states: (B, ...) tensor, where B is the number of state machines
        - actions: (B,) tensor
        """
        sorted_states, _ = torch.sort(states, dim=-1)
        _, inverse_indices, counts = torch.unique(sorted_states, dim=0, return_inverse=True, return_counts=True)
        unique_actions = torch.zeros((counts.shape[0],), device=actions.device, dtype=actions.dtype)
        _, first_pos = np.unique(inverse_indices.numpy(), return_index=True)
        mask = torch.zeros_like(inverse_indices, dtype=torch.bool)
        mask[first_pos] = True
        unique_actions.index_add_(0, inverse_indices[mask], actions[mask])
        forced_actions = unique_actions[inverse_indices]
        return forced_actions
