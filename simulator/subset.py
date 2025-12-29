from protocol.state import AbstractState, DummyState

from itertools import combinations

class SubsetManager:
    def __init__(self, num_state_machines: int, subset_size: int):
        self.all_subsets: list[Subset] = []
        self.lookup: dict[int, list[Subset]] = {}
        self.num_state_machines = num_state_machines
        self.subset_size = subset_size
        for comb in combinations(range(num_state_machines), subset_size):
            s = Subset(comb)
            self.all_subsets.append(s)
            for idx in comb:
                self.lookup.setdefault(idx, []).append(s)

    def init_states(self, states: list[AbstractState]):
        for subset in self.all_subsets:
            subset.do_transition([states[i] for i in subset.subset_ids])

    def get_final_states(self, idx: int) -> list[AbstractState]:
        final_states = []
        subsets = self.lookup.get(idx, [])
        for subset in subsets:
            subset_pos = subset.subset_ids.index(idx)
            final_states.append(subset.subset_states[subset_pos])
        return final_states

    def get_all_final_states(self) -> list[list[AbstractState]]:
        """
        Returns a list of final states for each state machine of all its subsets.
        """
        all_final_states = []
        for i in range(self.num_state_machines):
            all_final_states.append(self.get_final_states(i))
        return all_final_states

    def get_all_subset_states(self) -> list[list[AbstractState]]:
        """
        Returns a list of current all subset states.
        """
        return list(map(lambda s: s.subset_states, self.all_subsets))

    def get_msgs_with_mask(self, mask: list[list[int]]) -> list[list[AbstractState]]:
        """
        mask: n x [sender indices], can be seen as mailbox for each state machine

        Returns messages for all subsets based on the given mask.
        """
        lost_state = self.all_subsets[0].subset_states[0].get_lost_state()
        all_msgs = []
        for subset in self.all_subsets:
            for idx in subset.subset_ids:
                if mask[idx]:
                    subset_msgs = []
                    for i, sender in enumerate(subset.subset_ids):
                        subset_msgs.append(subset.subset_states[i] if sender in mask[idx] else lost_state)
                else:
                    subset_msgs = [lost_state] * self.subset_size
                all_msgs.append(subset_msgs)
        return all_msgs # n x subset_size list (each is msgs with length subset_size)

    def apply_states(self, states: list[AbstractState]):
        assert len(states) == len(self.all_subsets) * self.subset_size, "States length does not match subsets."
        idx = 0
        for subset in self.all_subsets:
            subset.do_transition(states[idx:idx + self.subset_size])
            idx += self.subset_size

class Subset:
    def __init__(self, ids):
        self.subset_ids: tuple[int] = ids
        self.subset_states: list[AbstractState] = []

    def __repr__(self):
        return str(self.subset_ids)

    def do_transition(self, next_states: list[AbstractState]):
        self.subset_states = next_states

if __name__ == "__main__":
    sm = SubsetManager(5, 3)
    sm.init_states([DummyState.Initial, DummyState.Intermediate, DummyState.Final, DummyState.Initial, DummyState.Final])
    print(sm.get_all_subset_states())
