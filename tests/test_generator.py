from simulator.generator import *

import math

def test_crash_node_generator():
    """ Test CrashNodeGenerator generates all possible crash patterns and message patterns correctly """

    num_nodes = range(3, 6)
    rounds = range(1, 4)

    def count_crash_patterns(N, round, crashed):
        return sum(math.comb(N, i) * (round ** i) for i in range(crashed + 1))

    for n in num_nodes:
        for r in rounds:
            gen = CrashNodeGenerator(num_nodes=n, rounds=r, last_round_work=True)
            assert len(gen.all_crash_patterns) == count_crash_patterns(n, r, r-1)
            assert len(gen.all_crash_patterns) == len(gen.all_message_patterns)
            gen = CrashNodeGenerator(num_nodes=n, rounds=r, last_round_work=False)
            assert len(gen.all_crash_patterns) == count_crash_patterns(n, r, n-1)
            assert len(gen.all_crash_patterns) == len(gen.all_message_patterns)


def test_initial_state_generator():
    """ Test InitialStateGenerator generates all possible initial states correctly """

    num_states = 2
    possible_initial_states = list(range(num_states))
    num_nodes = 5
    gen = InitialStateGenerator(num_nodes=num_nodes, legal_initial_state=possible_initial_states)

    # Verify the number of generated initial states
    expected_num_initial_states = len(possible_initial_states) ** num_nodes
    assert len(gen.all_initial_states) == expected_num_initial_states

    # Verify that each generated initial state is valid
    for initial_state in gen.all_initial_states:
        assert len(initial_state) == num_nodes
        for state in initial_state:
            assert state in possible_initial_states


def test_input_generator():
    """ Test various configurations of ReadableInputGenerator """

    """ Test 1: last_round_work = False, only 2 crashing nodes """
    num_players = 3
    num_rounds = 2 # can be modified
    num_states = 2 # can be modified
    # most crashing nodes = 2, which equals num_players - 1
    rig = ReadableInputGenerator(num_nodes=num_players, rounds=num_rounds, legal_initial_state=list(range(num_states)), last_round_work=False)
    rig.generate_all_inputs()
    # calculate the expected number of inputs
    crash_0 = 1
    crash_1 = math.comb(num_players, 1) * num_rounds * (2 ** (num_players - 1)) # which node crashes, which round, message sent or not to other nodes
    crash_2_same_round = num_rounds * (2 ** (num_players - 2) * 2 ** (num_players - 2))
    # all possible combinations of 2 nodes crashing in different rounds * message patterns
    crash_2_diff_round = math.prod([math.comb(num_rounds - i, 1) for i in range(2)]) * (2 ** (num_players - 1)) * (2 ** (num_players - 2))
    crash_2 = math.comb(num_players, 2) * (crash_2_same_round + crash_2_diff_round)
    expected_num_inputs = (num_states ** num_players) * (crash_0 + crash_1 + crash_2)
    assert len(rig.all_inputs) == expected_num_inputs, f"Expected {expected_num_inputs} inputs, but got {len(rig.all_inputs)}"

    """ Test 2: last_round_work = True, only 1 crashing node """
    num_players = 3
    num_rounds = 2
    num_states = 2 # can be modified
    # most crashing nodes = 1, which equals num_rounds - 1
    rig = ReadableInputGenerator(num_nodes=num_players, rounds=num_rounds, legal_initial_state=list(range(num_states)), last_round_work=True)
    rig.generate_all_inputs()
    crash_0 = 1
    crash_1 = math.comb(num_players, 1) * num_rounds * (2 ** (num_players - 1))
    expected_num_inputs = (num_states ** num_players) * (crash_0 + crash_1)
    assert len(rig.all_inputs) == expected_num_inputs, f"Expected {expected_num_inputs} inputs, but got {len(rig.all_inputs)}"

    """ Test 3: last_round_work = True, only 2 crashing node """
    num_players = 3
    num_rounds = 3 # can be modified to be >2
    num_states = 2 # can be modified
    # most crashing nodes = 2, which equals num_rounds - 1 or num_players - 1
    rig = ReadableInputGenerator(num_nodes=num_players, rounds=num_rounds, legal_initial_state=list(range(num_states)), last_round_work=True)
    rig.generate_all_inputs()
    crash_0 = 1
    crash_1 = math.comb(num_players, 1) * num_rounds * (2 ** (num_players - 1))
    crash_2_same_round = (num_rounds - 1) * (2 ** (num_players - 2) * 2 ** (num_players - 2)) \
        + (2 ** (num_players - 1) * 2 ** (num_players - 1)) # previous rounds + last round (which is different because of last_round_work)
    crash_2_diff_round = math.prod([math.comb(num_rounds - i, 1) for i in range(2)]) * (2 ** (num_players - 1)) * (2 ** (num_players - 2))
    crash_2 = math.comb(num_players, 2) * (crash_2_same_round + crash_2_diff_round)
    expected_num_inputs = (num_states ** num_players) * (crash_0 + crash_1 + crash_2)
    assert len(rig.all_inputs) == expected_num_inputs, f"Expected {expected_num_inputs} inputs, but got {len(rig.all_inputs)}"


def test_get_sender():
    """ Test CrashNodeGenerator.get_sender method """

    init_states = [0 for _ in range(4)]
    crash_pattern = [
        [],
        [0, 1]
    ]
    message_pattern = [
        np.array([], dtype=bool),
        np.array([[True, False], [False, True], [False, False], [False, True]])
    ]
    input_pattern = ReadableInput(
        initial_states=init_states,
        crash_pattern=crash_pattern,
        message_pattern=message_pattern
    )
    senders = get_sender_idx_from_input(input_pattern, round_idx=1, last_round_work=True)
    assert len(senders) == 4, f"Expected 4 senders, but got {len(senders)}"
    assert senders[0] == [0, 2, 3], f"Expected [0, 2, 3], but got {senders[0]}"
    assert senders[1] == [1, 2, 3], f"Expected [1, 2, 3], but got {senders[1]}"
    assert senders[2] == [2, 3], f"Expected [2, 3], but got {senders[2]}"
    assert senders[3] == [1, 2, 3], f"Expected [1, 2, 3], but got {senders[3]}"

    crash_pattern = [
        [],
        [1, 3]
    ]
    message_pattern = [
        np.array([], dtype=bool),
        np.array([[True, False], [False, True]])
    ]
    input_pattern = ReadableInput(
        initial_states=init_states,
        crash_pattern=crash_pattern,
        message_pattern=message_pattern
    )
    senders = get_sender_idx_from_input(input_pattern, round_idx=1, last_round_work=False)
    assert len(senders) == 4, f"Expected 4 senders, but got {len(senders)}"
    assert senders[0] == [0, 1, 2], f"Expected [0, 1, 2], but got {senders[0]}"
    assert senders[1] == [], f"Expected [], but got {senders[1]}"
    assert senders[2] == [0, 2, 3], f"Expected [0, 2, 3], but got {senders[2]}"
    assert senders[3] == [], f"Expected [], but got {senders[3]}"
