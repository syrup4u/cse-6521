import random

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