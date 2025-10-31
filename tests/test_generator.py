from simulator.generator import CrashNodeGenerator, InitialStateGenerator
from protocol.simple_majority import State

import math

def test_crash_node_generator():
    num_nodes = range(3, 6)
    rounds = range(1, 4)

    def count_crash_patterns(N, round, crashed):
        return sum(math.comb(N, i) * (round ** i) for i in range(crashed + 1))

    for n in num_nodes:
        for r in rounds:
            gen = CrashNodeGenerator(num_nodes=n, rounds=r, limit_crash=True)
            assert len(gen.all_crash_patterns) == count_crash_patterns(n, r, r-1)
            assert len(gen.all_crash_patterns) == len(gen.all_message_patterns)
            gen = CrashNodeGenerator(num_nodes=n, rounds=r, limit_crash=False)
            assert len(gen.all_crash_patterns) == count_crash_patterns(n, r, n-1)
            assert len(gen.all_crash_patterns) == len(gen.all_message_patterns)

def test_initial_state_generator():
    possible_initial_states = [State.NO, State.YES]
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
