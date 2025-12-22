from model.base import *

def test_policy_constraint_get_action():
    """ Test PolicyConstraint.get_action method """

    logits = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.3, 0.2, 0.1],
    ])
    states = torch.tensor([
        [1, 2],
        [2, 3],
        [2, 1],
        [3, 4],
        [3, 2],
    ])
    actions, log_probs = PolicyConstraint.get_action(logits, states)
    print("Actions:", actions)
    print("Log probabilities:", log_probs)
    assert actions[0] == actions[2], f"Expected actions[0] == actions[2], but got {actions[0]} != {actions[2]}"
    assert actions[1] == actions[4], f"Expected actions[1] == actions[4], but got {actions[1]} != {actions[4]}"
    assert log_probs[0] == log_probs[2], f"Expected log_probs[0] == log_probs[2], but got {log_probs[0]} != {log_probs[2]}"
    assert log_probs[1] == log_probs[4], f"Expected log_probs[1] == log_probs[4], but got {log_probs[1]} != {log_probs[4]}"

def test_policy_constraint_force_same_action():
    """ Test PolicyConstraint.force_same_action method """

    actions = torch.tensor([0, 1, 2, 3, 4])
    states = torch.tensor([
        [1, 2],
        [2, 3],
        [2, 1],
        [3, 4],
        [3, 2],
    ])
    forced_actions = PolicyConstraint.force_same_action(actions, states)
    print("Forced Actions:", forced_actions)
    assert torch.tensor([0, 1, 0, 3, 1]).equal(forced_actions), f"Expected forced actions to be [0, 1, 0, 3, 1], but got {forced_actions.tolist()}"
