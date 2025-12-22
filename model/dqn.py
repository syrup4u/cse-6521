from .base import *
import config

import torch
from torch import nn
from torch import optim
import logging
from typing import Optional
import collections
import random

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    
    def __len__(self):
        return len(self.buffer)


class DeepQNetwork(nn.Module):
    def __init__(self, n_actions, q_net: nn.Module, target: Optional[nn.Module] = None):
        super().__init__()
        self.n_actions = n_actions
        self.q_net = q_net
        self.target_net = target if target is not None else q_net
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.double_q = target is not None
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.DQN_CONFIG["learning_rate"])
        self.loss_fn = getattr(nn, config.DQN_CONFIG["loss"])()
        self.target_update_freq = config.DQN_CONFIG["target_update_freq"]
        self.eps_cfg = {
            "start": config.DQN_CONFIG["eps_start"],
            "end": config.DQN_CONFIG["eps_end"],
            "decay": config.DQN_CONFIG["eps_decay"],
            "current": config.DQN_CONFIG["eps_start"]
        }
        self.episode = 0
        self.learn_step = 0
        # Replay buffer
        self.replay = ReplayBuffer(config.DQN_CONFIG["buffer_size"])
        self.batch_size = config.DQN_CONFIG["batch_size"]
        # Additional components
        self.input_encoder = None
        self.constraint = None
    
    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets action and Q-value of the action.
        - state: (B, ...) tensor
        - returns: (action tensor of shape (B,), Q-value tensor of shape (B,))
        - B is the number of state machines or stacked states

        v1: implement enforcement for consistent action among agents with same state.
        """
        if random.random() < self.eps_cfg["current"]: # Explore
            action = torch.randint(0, self.n_actions, (state.shape[0],), device=state.device)
        q_values = self.q_net(self.input_encoder(state)) # (B, n_actions)
        action = torch.argmax(q_values, dim=-1)
        if self.constraint is not None:
            action = self.constraint(action, state)
        # q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action, None # (B,), (B,)

    def get_greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        q_values = self.q_net(self.input_encoder(state)) # (B, n_actions)
        action = torch.argmax(q_values, dim=-1)
        return action  # (B,)

    def store_transition(self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor
    ):
        self.replay.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        
        # Sample history
        transitions = self.replay.sample(self.batch_size)
        states = torch.cat(transitions.state, dim=0)          # (B, ...)
        actions = torch.cat(transitions.action, dim=0)        # (B,)
        rewards = torch.cat(transitions.reward, dim=0)        # (B,)
        next_states = torch.cat(transitions.next_state, dim=0)  # (B, ...)
        dones = torch.cat(transitions.done, dim=0)            # (B,)

        # Current Q
        q_values = self.q_net(self.input_encoder(states)).gather(1, actions.unsqueeze(-1)).squeeze(-1)  # (B,)

        # Compute target Q
        with torch.no_grad():
            # next_q_online = self.q_net(self.input_encoder(next_states))  # (B, n_actions)
            # next_actions = torch.argmax(next_q_online, dim=-1)  # (B,)
            # next_q_target = self.target_net(self.input_encoder(next_states)).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)  # (B,)
            # q_values_target = rewards + (1 - dones.float()) * next_q_target  # (B,)
            q_values_target = rewards # Only related to the final reward (degrade to Monte Carlo)
        
        # Compute loss
        loss = self.loss_fn(q_values, q_values_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()


    def update_hyper(self, episodes: int):
        self.episode += episodes
        # Update epsilon
        if self.episode <= self.eps_cfg["decay"]:
            self.eps_cfg["current"] = max(
                self.eps_cfg["end"],
                self.eps_cfg["start"] - (self.eps_cfg["start"] - self.eps_cfg["end"]) * (self.episode / self.eps_cfg["decay"])
            )

    def save_model(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint)
        self.target_net.load_state_dict(checkpoint)

def build_mlp_model(
        input_size: int,
        output_size: int,
        device: str = 'cpu'
    ) -> DeepQNetwork:
    q_net = MLP(input_size, config.MLP_CONFIG["hidden_sizes"], output_size).to(device)
    target_net = MLP(input_size, config.MLP_CONFIG["hidden_sizes"], output_size).to(device)
    model = DeepQNetwork(output_size, q_net, target_net)
    model.input_encoder = lambda x: x.float()
    return model

def build_mlp_op_model(
        one_hot_length: int,
        output_size: int,
        device: str = 'cpu'
    ) -> DeepQNetwork:
    """
    - one_hot_length: length of one-hot encoded state representation, which is equal to
        (state space size + num_rounds)
    """
    q_net = MLP(one_hot_length, config.MLP_CONFIG["hidden_sizes"], output_size).to(device)
    target_net = MLP(one_hot_length, config.MLP_CONFIG["hidden_sizes"], output_size).to(device)
    model = DeepQNetwork(output_size, q_net, target_net)
    model.input_encoder = q_net.encode_sequence
    model.constraint = PolicyConstraint.force_same_action
    return model

def build_set_transformer_model(
        dim_output: int,
        num_states: int,
        num_rounds: int,
        device: str = 'cpu'
    ) -> DeepQNetwork:
    q_net = SetTransformer(
        dim_input=config.SET_TRANSFORMER_CONFIG["dim_input"],
        dim_output=dim_output,
        num_inds=config.SET_TRANSFORMER_CONFIG["num_inds"],
        dim_hidden=config.SET_TRANSFORMER_CONFIG["dim_hidden"],
        num_heads=config.SET_TRANSFORMER_CONFIG["num_heads"],
        num_outputs=config.SET_TRANSFORMER_CONFIG["num_outputs"],
        num_states=num_states,
        num_rounds=num_rounds,
        encode_round_number=config.ENCODE_ROUND_NUMBER
    ).to(device)
    target_net = SetTransformer(
        dim_input=config.SET_TRANSFORMER_CONFIG["dim_input"],
        dim_output=dim_output,
        num_inds=config.SET_TRANSFORMER_CONFIG["num_inds"],
        dim_hidden=config.SET_TRANSFORMER_CONFIG["dim_hidden"],
        num_heads=config.SET_TRANSFORMER_CONFIG["num_heads"],
        num_outputs=config.SET_TRANSFORMER_CONFIG["num_outputs"],
        num_states=num_states,
        num_rounds=num_rounds,
        encode_round_number=config.ENCODE_ROUND_NUMBER
    ).to(device)
    model = DeepQNetwork(dim_output, q_net, target_net)
    model.input_encoder = q_net.tok_emb
    model.constraint = PolicyConstraint.force_same_action
    return model

def train_model(model: DeepQNetwork, trajectories: dict, rewards: list, others=None):
    """
    - traj_states: (B) batch of trajectories of states, each trajectory is a list of (R, N, ...) tensors
    - traj_actions: (B) batch of trajectories of actions, each trajectory is a list of (R, N) tensors
    - rewards: list of rewards corresponding to each trajectory in batch, which is of length B (batch size)
    """
    # Prepare training data
    traj_state = trajectories["states"]
    traj_actions = trajectories["actions"]
    traj_done = trajectories["done"]
    B = len(traj_state) # batch size
    R = len(traj_state[0]) # number of rounds
    
    # Update replay buffer
    for b in range(B):
        for r in range(R):
            state = traj_state[b][r] # (N, ...) tensor, N is number of alive nodes (may vary across rounds)
            action = traj_actions[b][r]
            done = traj_done[b][r]
            next_state = torch.zeros_like(state) # ensure same shape
            if r < R - 1:
                if traj_state[b][r+1].shape[0] != state.shape[0]:
                    next_state[~done] = traj_state[b][r+1] # fill in next state for non-done nodes
                else: # It is possible that a next-round-crashed node still has a next state (at last round)
                    next_state = traj_state[b][r+1]
            reward = torch.full_like(action, rewards[b])
            done = done.float()
            model.store_transition(state, action, reward, next_state, done)
    
    for _ in range(B):
        loss = model.update()
    model.update_hyper(others["episodes"])
    
    logger.debug(f"Training loss: {loss}")


def update_trajectories(
        trajectories: dict,
        state: torch.Tensor = None,
        action: torch.Tensor = None,
        others: list = None
    ):
    if "one_episode" not in trajectories:
        trajectories["one_episode"] = dict()
    if state is not None:
        trajectories["one_episode"].setdefault("states", []).append(state) # add R dimension
        trajectories["one_episode"].setdefault("actions", []).append(action)
        trajectories["one_episode"].setdefault("done", []).append(torch.tensor(others[1], device=state.device)) # done flag
    else:
        trajectories.setdefault("states", []).append(trajectories["one_episode"]["states"]) # add B dimension
        trajectories.setdefault("actions", []).append(trajectories["one_episode"]["actions"])
        trajectories.setdefault("done", []).append(trajectories["one_episode"]["done"])
        trajectories["one_episode"] = dict()