import random
import config

def sample_from_two_lists(list1, list2, probability=0.1):
    """
    Samples an element from list1 with the given probability,
    otherwise samples from list2.
    """
    r = random.random()
    if r < probability:
        return random.choice(list1)
    else:
        return random.choice(list2)

def get_round_info(current_round: int, total_rounds: int) -> int:
    """
    Returns the current round number and whether it's the last round.
    """
    if config.ENCODE_ROUND_NUMBER:
        return current_round
    else:
        return int(current_round == total_rounds - 1)
