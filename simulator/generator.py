from protocol.state import AbstractState
from protocol.simple_majority import State

from itertools import combinations, product
import numpy as np
from typing import List

class CrashNodeGenerator:
    """
    Generates all possible crash node patterns and corresponding message patterns for a given number of nodes and rounds.
    """
    
    def __init__(self, num_nodes: int, rounds: int, *, limit_crash: bool = True):
        """
        <limit_crash>: if True, at most rounds - 1 nodes can crash in total.
            True reason is that the nodes crashing in the last round could still make decisions.
        """

        self.num_nodes = num_nodes
        self.rounds = rounds
        self.all_crash_patterns = None
        self.all_message_patterns = None

        # TODO: can use streams to yield patterns instead of storing all in memory
        # if needed (for large num_nodes, rounds), by saving a Tree structure
        most_crash = min(rounds - 1, num_nodes - 1) if limit_crash else num_nodes - 1
        self.generate_all_crash(most_crash)
        self.generate_all_message_for_crash(last_round_work=limit_crash)

    def generate_all_crash(self, most_crash: int):
        """
        Generates all possible crash patterns.
        And assigns them to each round without contradiction.
        """

        """
        each crash pattern is a list of sets, where each set contains the indices of nodes that crash in that round.
        Example:
        [
            {0},        # round 0: node 0 crashes
            {1, 2},     # round 1: nodes 1 and 2 crash
            {}          # round 2: no crashes
        ]
        """
        self.all_crash_patterns = []

        def dfs(round_idx: int, remaining_nodes: list, crash_so_far: set, crash_pattern: list):
            if round_idx == self.rounds:
                self.all_crash_patterns.append([round_crash for round_crash in crash_pattern])
                return
            if len(crash_so_far) == most_crash:
                dfs(round_idx + 1, remaining_nodes, crash_so_far, crash_pattern + [set()])
                return
            max_to_crash = min(len(remaining_nodes), most_crash - len(crash_so_far))
            for num_crash in range(0, max_to_crash + 1):
                for nodes_to_crash in combinations(remaining_nodes, num_crash):
                    new_crash_so_far = crash_so_far.union(set(nodes_to_crash))
                    new_remaining_nodes = [n for n in remaining_nodes if n not in nodes_to_crash]
                    dfs(round_idx + 1, new_remaining_nodes, new_crash_so_far, crash_pattern + [set(nodes_to_crash)])

        dfs(0, list(range(self.num_nodes)), set(), [])


    def generate_all_message_for_crash(self, last_round_work: bool = True):
        """
        Generates all possible message patterns according to crash patterns.
        Uses masks to represent lost / sent messages.
        <last_round_work>: even if nodes crash in the last round, they can still receive messages and make decisions. (a synchronous model assumption)
        """

        assert self.all_crash_patterns is not None, "Crash patterns must be generated before message patterns."

        """
        Corresponding to self.all_crash_patterns.
        Each message pattern is a list of lists of NumPy bool arrays.
        Each list corresponds to a round, and contains all possible arrays of message masks.
        Each array has shape (num_receivers, num_senders), where each element is True if the message is received, False if lost.
        Receivers are alive nodes at that round, senders are new crashed nodes at that round.
        """
        self.all_message_patterns = []

        def get_receiver_row_patterns(senders: List[int], receiver: int) -> List[np.ndarray]:
            """
            Return a *list* of NumPy bool arrays, each of length |senders|,
            representing every legal bit-row for this receiver.

            If the receiver itself is one of the just-crashed senders, its
            own bit (diagonal) is forced to 1.

            Example:
            get_receiver_row_patterns(senders = [0, 1], receiver = 0):
            [array([ True, False]), array([ True,  True])]
            Meaning:
            Node 0 and 1 newly crash, node 0 receives has its own message, and can or can not receive node 1's message.
            """
            n = len(senders)
            if receiver not in senders:
                return [
                    np.array(bits, dtype=bool)  # -> shape (n,)
                    for bits in product([0, 1], repeat=n)
                ]
            
            self_idx = senders.index(receiver)
            patterns = []
            for bits in product([0, 1], repeat=n-1):
                vec = list(bits)
                vec.insert(self_idx, 1)  # force diag = 1
                patterns.append(np.array(vec, dtype=bool))
            return patterns # -> list of arrays of shape (n,)
        
        for crash_pattern in self.all_crash_patterns:
            # message patterns for this crash pattern
            message_patterns = []
            crash_so_far = set()
            for round_idx, crashed_nodes in enumerate(crash_pattern):
                senders = sorted(list(crashed_nodes))
                if round_idx == self.rounds - 1 and last_round_work:
                    receivers = [n for n in range(self.num_nodes) if n not in crash_so_far]
                else:
                    receivers = [n for n in range(self.num_nodes) if n not in crash_so_far.union(crashed_nodes)]
                crash_so_far = crash_so_far.union(crashed_nodes)
                row_patterns_per_receiver = [get_receiver_row_patterns(senders, receiver) for receiver in receivers]
                all_cases_this_round = []
                for one_case in product(*row_patterns_per_receiver):
                    # np shape: (num_receivers, num_senders)
                    all_cases_this_round.append(np.stack(one_case, axis=0))
                message_patterns.append(all_cases_this_round)
            
            self.all_message_patterns.append(message_patterns)


class InitialStateGenerator:
    """ Generates all possible initial states for the nodes. """
    
    def __init__(self, num_nodes: int, legal_initial_state: List[AbstractState]):
        self.num_nodes = num_nodes
        self.legal_initial_state = legal_initial_state
        self.all_initial_states: List[List[AbstractState]] = None
        self.generate_all_state()

    def generate_all_state(self):
        """
        Generates all possible combinations of initial states for the nodes.
        Each combination is a list of States of length num_nodes.
        """

        self.all_initial_states = []

        for initial_states in product(self.legal_initial_state, repeat=self.num_nodes):
            self.all_initial_states.append(list(initial_states))


class ReadableInput:
    """
    A readable representation of one possible input combination, combining crash patterns, message patterns, and initial states.
    """

    def __init__(self, initial_states: List[AbstractState], crash_pattern: List[set], message_pattern: List[np.ndarray]):
        self.initial_states = initial_states
        self.crash_pattern = crash_pattern
        self.message_pattern = message_pattern

    def __str__(self):
        result = []
        result.append(f"Initial States: {self.initial_states}")
        for round_idx, (crashed_nodes, msg_mask) in enumerate(zip(self.crash_pattern, self.message_pattern)):
            result.append(f"Round {round_idx + 1}")
            result.append(f"-- Crashed Nodes: {crashed_nodes}")
            result.append(f"-- Message Mask:\n{msg_mask}")

        return "\n".join(result)


class ReadableInputGenerator:
    """
    Combines CrashNodeGenerator and InitialStateGenerator to produce readable input patterns.
    """

    def __init__(self, num_nodes: int, rounds: int, *, legal_initial_state: List[AbstractState], limit_crash: bool = True):
        self.gen_crash = CrashNodeGenerator(num_nodes=num_nodes, rounds=rounds, limit_crash=limit_crash)
        self.gen_initial = InitialStateGenerator(num_nodes=num_nodes, legal_initial_state=legal_initial_state)
        self.all_inputs = None

    def generate_all_inputs(self):
        self.all_inputs = []
        for initial_states in self.gen_initial.all_initial_states:
            for crash_idx, crash_pattern in enumerate(self.gen_crash.all_crash_patterns):
                message_patterns = self.gen_crash.all_message_patterns[crash_idx]
                # crash_pattern and message_patterns have r rounds
                for msg_pattern_combination in product(*message_patterns):
                    self.all_inputs.append(ReadableInput(
                        initial_states=initial_states,
                        crash_pattern=crash_pattern,
                        message_pattern=msg_pattern_combination
                    ))

    def generate_yield_inputs(self):
        """ A generator version of generate_all_inputs."""
        for initial_states in self.gen_initial.all_initial_states:
            for crash_idx, crash_pattern in enumerate(self.gen_crash.all_crash_patterns):
                message_patterns = self.gen_crash.all_message_patterns[crash_idx]
                # crash_pattern and message_patterns have r rounds
                for msg_pattern_combination in product(*message_patterns):
                    yield ReadableInput(
                        initial_states=initial_states,
                        crash_pattern=crash_pattern,
                        message_pattern=msg_pattern_combination
                    )


if __name__ == "__main__":
    # gen = CrashNodeGenerator(num_nodes=3, rounds=2, limit_crash=True)
    # for pattern in gen.all_crash_patterns:
    #     print(pattern)
    # print(len(gen.all_crash_patterns))
    # cnt = 0
    # for msg_patterns in gen.all_message_patterns:
    #     cnt_for_this_patterns = 1
    #     for round_idx, round_patterns in enumerate(msg_patterns):
    #         cnt_for_this_patterns *= len(round_patterns)
    #     cnt += cnt_for_this_patterns
    # print(cnt)
    gen = ReadableInputGenerator(num_nodes=3, rounds=2, legal_initial_state=[State.NO, State.YES], limit_crash=True)
    cnt = 0
    for input_pattern in gen.generate_yield_inputs():
        # print(input_pattern)
        # print("-----")
        cnt += 1
    print(f"Total input patterns: {cnt}")
