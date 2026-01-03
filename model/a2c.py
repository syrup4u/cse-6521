from .base import *
import config

import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class ActorCritic(nn.Module):
    def __init__(self, actor: nn.Module, critic: nn.Module, learning_rate: float):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        # Additional components
        self.input_encoder = None
        self.constraint = None

    def get_logits(self, state: torch.Tensor) -> torch.Tensor:
        if self.input_encoder is not None:
            state = self.input_encoder(state)
        return self.actor(state)

    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets action and log probability of the action.
        - state: (B, N+1) tensor
        - returns: (action tensor of shape (B,), log_prob tensor of shape (B,))
        - B is the number of state machines, N+1 is the length of the state sequence (including round number)

        v1: implement enforcement for consistent action among agents with same state.
        """
        logits = self.get_logits(state)
        if self.constraint is None:
            m = Categorical(logits=logits)
            action = m.sample()
            probs = m.log_prob(action)
        else:
            action, probs = self.constraint(logits, state)
        return action, probs # shape: (B,)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        if self.input_encoder is not None:
            state = self.input_encoder(state)
        value = self.critic(state) # shape: (B, 1)
        return value.squeeze(-1) # shape: (B,)

    def get_greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.get_logits(state)
        action = torch.argmax(logits, dim=-1)
        return action
    
    def update_hyper(self, episode: int):
        pass
    
    def save_model(self, path: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


def build_mlp_model(
        cfg: config.Config,
        input_size: int,
        output_size: int,
        device: str = 'cpu'
    ) -> ActorCritic:
    actor = MLP(input_size, cfg.model.mlp.hidden_sizes, output_size).to(device)
    critic = MLP(input_size, cfg.model.mlp.hidden_sizes, 1).to(device)
    model = ActorCritic(actor, critic, cfg.algorithm.learning_rate)
    model.input_encoder = lambda x: x.float()
    return model

def build_mlp_op_model(
        cfg: config.Config,
        one_hot_length: int,
        output_size: int,
        device: str = 'cpu'
    ) -> ActorCritic:
    """
    - one_hot_length: length of one-hot encoded state representation, which is equal to
        (state space size + num_rounds)
    """
    actor = MLP(one_hot_length, cfg.model.mlp.hidden_sizes, output_size).to(device)
    critic = MLP(one_hot_length, cfg.model.mlp.hidden_sizes, 1).to(device)
    model = ActorCritic(actor, critic, cfg.algorithm.learning_rate)
    model.input_encoder = actor.encode_sequence
    model.constraint = PolicyConstraint.get_action
    return model

def build_set_transformer_model(
        cfg: config.Config,
        dim_output: int,
        num_states: int,
        num_rounds: int,
        device: str = 'cpu'
    ) -> ActorCritic:
    actor = SetTransformer(
        dim_input=cfg.model.st.dim_input,
        dim_output=dim_output,
        num_inds=cfg.model.st.num_inds,
        dim_hidden=cfg.model.st.dim_hidden,
        num_heads=cfg.model.st.num_heads,
        num_outputs=cfg.model.st.num_outputs,
        num_states=num_states,
        num_rounds=num_rounds,
        encode_round_number=cfg.model.encode_round_number
    ).to(device)
    critic = SetTransformer(
        dim_input=cfg.model.st.dim_input,
        dim_output=1,
        num_inds=cfg.model.st.num_inds,
        dim_hidden=cfg.model.st.dim_hidden,
        num_heads=cfg.model.st.num_heads,
        num_outputs=cfg.model.st.num_outputs,
        num_states=num_states,
        num_rounds=num_rounds,
        encode_round_number=cfg.model.encode_round_number
    ).to(device)
    model = ActorCritic(actor, critic, cfg.algorithm.learning_rate)
    model.input_encoder = actor.tok_emb
    model.constraint = PolicyConstraint.get_action
    return model

def train_model(cfg: config.Config, model: ActorCritic, trajectories: dict, rewards: list, others=None):
    """
    - traj_states: (B) batch of trajectories of states, each trajectory is a list of (R, N, ...) tensors
    - traj_actions: (B) batch of trajectories of actions, each trajectory is a list of (R, N) tensors
    - traj_log_probs: (B) batch of trajectories of log_probs, each trajectory is a list of (R, N) tensors
    - rewards: list of rewards corresponding to each trajectory in batch, which is of length B (batch size)

    B: batch size,
    R: number of rounds,
    N: number of nodes
    """
    traj_state = trajectories["states"]
    traj_log_probs = trajectories["log_probs"]
    traj_actions = trajectories["actions"]
    traj_dones = trajectories["done"]
    B = len(traj_state) # batch size
    R = len(traj_state[0]) # number of rounds
    N = traj_state[0][R-1].shape[0] # number of valid nodes (may vary among trajectories)

    # Preprocess trajectories
    B_states = []
    B_actions = []
    B_log_probs = []
    for b in range(B):
        b_states = None
        b_actions = None
        b_log_probs = None
        for r in range(R):
            if r < R - 2:
                mask = ~traj_dones[b][r]
                reduced_states = traj_state[b][r][mask]
                reduced_actions = traj_actions[b][r][mask]
                reduced_log_probs = traj_log_probs[b][r][mask]
                if b_states is not None:
                    b_states = b_states[:, mask, ...]
                    b_actions = b_actions[:, mask, ...]
                    b_log_probs = b_log_probs[:, mask, ...]
            else:
                reduced_states = traj_state[b][r]
                reduced_actions = traj_actions[b][r]
                reduced_log_probs = traj_log_probs[b][r]
            if b_states is None:
                b_states = reduced_states.unsqueeze(0)
                b_actions = reduced_actions.unsqueeze(0)
                b_log_probs = reduced_log_probs.unsqueeze(0)
            else:
                b_states = torch.cat((b_states, reduced_states.unsqueeze(0)), dim=0)
                b_actions = torch.cat((b_actions, reduced_actions.unsqueeze(0)), dim=0)
                b_log_probs = torch.cat((b_log_probs, reduced_log_probs.unsqueeze(0)), dim=0)
        B_states.append(b_states)
        B_actions.append(b_actions)
        B_log_probs.append(b_log_probs)

    all_states = torch.stack(B_states, dim=0) # (batch_size, R, N, ...)
    all_actions = torch.stack(B_actions, dim=0) # (batch_size, R, N)
    all_log_probs = torch.stack(B_log_probs, dim=0) # (batch_size, R, N)
    all_values = model.get_value(all_states.reshape(-1, all_states.shape[-1])) # (batch_size * R,)
    all_values = all_values.reshape(B, R, N) # (batch_size, R, N)
    all_R = torch.tensor(rewards, device=all_values.device, dtype=all_values.dtype)\
        .unsqueeze(1).unsqueeze(2).expand(-1, R, N) # (batch_size, R, N)

    advantages = compute_advantages(all_R, all_values, gamma=1.0) # (batch_size, R, N)
    returns = advantages + all_values # (batch_size, R, N)
    # oversampling non-zero reward
    # oversampling_weight = 2 ** torch.abs(all_R)
    # advantages = oversampling_weight * advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    all_log_probs = all_log_probs.detach()
    returns = returns.detach()
    advantages = advantages.detach()

    # PPO updates
    EPS_CLIP = cfg.algorithm.a2c.clip_epsilon
    entropy_gamma = cfg.algorithm.a2c.entropy_gamma
    ppo_epochs = cfg.algorithm.a2c.ppo_epochs
    for _ in range(ppo_epochs):
        new_logits = model.get_logits(all_states.reshape(-1, all_states.shape[-1])) # (batch_size * R * N, num_actions)
        m = Categorical(logits=new_logits)
        new_log_probs = m.log_prob(all_actions.reshape(-1)) # (batch_size * R * N,)
        new_log_probs = new_log_probs.reshape(B, R, N) # (batch_size, R, N)
        new_values = model.get_value(all_states.reshape(-1, all_states.shape[-1])) # (batch_size * R,)
        new_values = new_values.reshape(B, R, N) # (batch_size, R, N)

        ratio = torch.exp(new_log_probs - all_log_probs) # (batch_size, R, N)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * advantages

        # Update actor and critic
        entropy_bonus = entropy_gamma * m.entropy().mean() # Entropy bonus
        actor_loss = -torch.mean(torch.min(surr1, surr2)) - entropy_bonus # PPO clipped objective
        critic_loss = nn.functional.mse_loss(new_values, returns) # Value function loss

        # Backpropagation for actor
        model.actor_optimizer.zero_grad()
        actor_loss.backward()
        model.actor_optimizer.step()
        # Backpropagation for critic
        model.critic_optimizer.zero_grad()
        critic_loss.backward()
        model.critic_optimizer.step()

    logger.debug(f"ratio mean: {torch.mean(ratio).item()}")
    logger.debug(f"Batch Actor loss: {actor_loss.item()}, Batch Critic loss: {critic_loss.item()}")
    logger.debug(f"Entropy bonus: {entropy_bonus.item()}")
    logger.debug(f"Probs: {torch.exp(new_log_probs).mean().item()}")

def compute_advantages(rewards: torch.Tensor, values: torch.Tensor, gamma=0.99, lam=0.95) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE).
    - rewards & values: (B, R, N) tensor, where B is batch size, R is number of rounds, N is number of nodes
    """
    B, R, N = rewards.shape
    advantages = torch.zeros((B, R, N), device=rewards.device)

    for t in reversed(range(R)):
        delta = rewards[:, t, :] + gamma * (values[:, t + 1, :] if t + 1 < R else 0) - values[:, t, :]
        advantages[:, t, :] = delta + gamma * lam * (advantages[:, t + 1, :] if t + 1 < R else 0)

    return advantages

def update_trajectories(
        trajectories: dict,
        state: torch.Tensor = None,
        action: torch.Tensor = None,
        others: list = None
    ):
    if "one_episode" not in trajectories:
        trajectories["one_episode"] = dict()
    if state is not None:
        trajectories["one_episode"].setdefault("states", []).append(state)
        trajectories["one_episode"].setdefault("actions", []).append(action)
        trajectories["one_episode"].setdefault("log_probs", []).append(others[0])
        trajectories["one_episode"].setdefault("done", []).append(torch.tensor(others[1], device=state.device))
    else:
        trajectories.setdefault("states", []).append(trajectories["one_episode"]["states"])
        trajectories.setdefault("actions", []).append(trajectories["one_episode"]["actions"])
        trajectories.setdefault("log_probs", []).append(trajectories["one_episode"]["log_probs"])
        trajectories.setdefault("done", []).append(trajectories["one_episode"]["done"])
        trajectories["one_episode"] = dict()
