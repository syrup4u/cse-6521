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
        return self.network(x) # (B, output_size)

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

        # TODO: may add an extra token embedding for num_rounds (not just round number)
        self.tok_emb = nn.Embedding(num_states + num_rounds, dim_input) # B x (N+1) -> x D

        # Mark: do we need to tie the embedding weights with the output layer weights? No.
        # self.dec[-1].weight = self.tok_emb.weight
    
    def forward(self, X):
        return self.dec(self.enc(X)).squeeze(-1) # (B, dim_output)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
