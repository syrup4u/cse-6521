from .base import *
import config

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
import logging

logger = logging.getLogger(__name__)

class ActorCritic:
    def __init__(self, actor: nn.Module, critic: nn.Module, learning_rate: float):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.input_encoder = None

    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets action and log probability of the action.
        - state: (B, ...) tensor
        - returns: (action tensor of shape (B,), log_prob tensor of shape (B,))

        TODO: do we need to implement enforcement for consistent action among agents with same state?
        """
        if self.input_encoder is not None:
            state = self.input_encoder(state)
        logits = self.actor(state)
        m = Categorical(logits=logits)
        action = m.sample()
        return action, m.log_prob(action) # shape: (B,)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        if self.input_encoder is not None:
            state = self.input_encoder(state)
        value = self.critic(state)
        return value.squeeze(-1) # shape: (B,)

    def get_greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        if self.input_encoder is not None:
            state = self.input_encoder(state)
        logits = self.actor(state)
        action = torch.argmax(logits, dim=-1)
        return action

def build_mlp_model(
        input_size: int,
        output_size: int,
        device: str = 'cpu'
    ) -> ActorCritic:
    actor = MLP(input_size, config.MLP_CONFIG["hidden_sizes"], output_size).to(device)
    critic = MLP(input_size, config.MLP_CONFIG["hidden_sizes"], 1).to(device)
    model = ActorCritic(actor, critic, config.MLP_CONFIG["learning_rate"])
    model.input_encoder = lambda x: x.float()
    return model

def build_mlp_op_model(
        one_hot_length: int,
        output_size: int,
        device: str = 'cpu'
    ) -> ActorCritic:
    """
    - one_hot_length: length of one-hot encoded state representation, which is equal to
        (state space size + num_rounds)
    """
    actor = MLP(one_hot_length, config.MLP_CONFIG["hidden_sizes"], output_size).to(device)
    critic = MLP(one_hot_length, config.MLP_CONFIG["hidden_sizes"], 1).to(device)
    model = ActorCritic(actor, critic, config.MLP_CONFIG["learning_rate"])
    model.input_encoder = actor.encode_sequence
    return model

def build_set_transformer_model(
        dim_output: int,
        num_states: int,
        num_rounds: int,
        device: str = 'cpu'
    ) -> ActorCritic:
    actor = SetTransformer(
        dim_input=config.SET_TRANSFORMER_CONFIG["dim_input"],
        dim_output=dim_output,
        num_inds=config.SET_TRANSFORMER_CONFIG["num_inds"],
        dim_hidden=config.SET_TRANSFORMER_CONFIG["dim_hidden"],
        num_heads=config.SET_TRANSFORMER_CONFIG["num_heads"],
        num_outputs=config.SET_TRANSFORMER_CONFIG["num_outputs"],
        num_states=num_states,
        num_rounds=num_rounds
    ).to(device)
    critic = SetTransformer(
        dim_input=config.SET_TRANSFORMER_CONFIG["dim_input"],
        dim_output=1,
        num_inds=config.SET_TRANSFORMER_CONFIG["num_inds"],
        dim_hidden=config.SET_TRANSFORMER_CONFIG["dim_hidden"],
        num_heads=config.SET_TRANSFORMER_CONFIG["num_heads"],
        num_outputs=config.SET_TRANSFORMER_CONFIG["num_outputs"],
        num_states=num_states,
        num_rounds=num_rounds
    ).to(device)
    model = ActorCritic(actor, critic, config.SET_TRANSFORMER_CONFIG["learning_rate"])
    model.input_encoder = actor.tok_emb
    return model

def train_model(model: ActorCritic, trajectories: list, reward: int):
    """
    Current implementation: just backpropagate after each episode.
    - trajectories: list of (state_tensor, actions, log_probs, values) tuples, length = R (number of rounds)

    - state_tensor: (B, N, D=1) tensor
    - actions: (B,) tensor
    - log_probs: (B,) tensor
    - values: (B,) tensor
    """
    # Compute returns and advantages
    all_log_probs = torch.stack([t[2] for t in trajectories], dim=0) # (R, B)
    all_values = torch.stack([t[3] for t in trajectories], dim=0) # (R, B)
    R = torch.full_like(all_values, fill_value=reward).detach() # (R, B), No discounting
    advantages = R - all_values # (R, B)

    # Update actor and critic
    actor_loss = 0
    critic_loss = 0
    actor_loss += -torch.mean(all_log_probs * advantages.detach()) # Policy gradient loss
    critic_loss += nn.functional.mse_loss(all_values, R) # Value function loss

    # Backpropagation for actor
    model.actor_optimizer.zero_grad()
    actor_loss.backward()
    model.actor_optimizer.step()

    # Backpropagation for critic
    model.critic_optimizer.zero_grad()
    critic_loss.backward()
    model.critic_optimizer.step()

    logger.debug(f"Actor loss: {actor_loss.item()}, Critic loss: {critic_loss.item()}")

def train_model_batch(model: ActorCritic, batch_trajectories: list, rewards: list):
    """
    - batch_trajectories: list of trajectories, each trajectory is a list of (state_tensor, actions, log_probs, values) tuples
    - rewards: list of rewards corresponding to each trajectory
    """
    all_log_probs = []
    all_values = []
    all_R = []

    for traj, reward in zip(batch_trajectories, rewards):
        log_probs = torch.stack([t[2] for t in traj], dim=0) # (R, B)
        values = torch.stack([t[3] for t in traj], dim=0) # (R, B)
        R = torch.full_like(values, fill_value=reward).detach() # (R, B)

        all_log_probs.append(log_probs)
        all_values.append(values)
        all_R.append(R)

    all_log_probs = torch.cat(all_log_probs, dim=1) # (R, total_B)
    all_values = torch.cat(all_values, dim=1) # (R, total_B)
    all_R = torch.cat(all_R, dim=1) # (R, total_B)

    advantages = all_R - all_values # (R, total_B)
    # oversampling non-zero reward
    non_zero_mask = (all_R != 0)
    oversampling_weight = 5
    advantages = oversampling_weight * advantages * non_zero_mask + advantages * (~non_zero_mask)

    # Update actor and critic
    actor_loss = 0
    critic_loss = 0
    actor_loss += -torch.mean(all_log_probs * advantages.detach()) # Policy gradient loss
    critic_loss += nn.functional.mse_loss(all_values, all_R) # Value function loss

    # Backpropagation for actor
    model.actor_optimizer.zero_grad()
    actor_loss.backward()
    model.actor_optimizer.step()

    # Backpropagation for critic
    model.critic_optimizer.zero_grad()
    critic_loss.backward()
    model.critic_optimizer.step()

    logger.debug(f"Batch Actor loss: {actor_loss.item()}, Batch Critic loss: {critic_loss.item()}")