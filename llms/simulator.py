from .z3_verifier import Z3Verifier

from importlib import import_module
import logging
import z3

logger = logging.getLogger(__name__)

class LLMProtocolSimulator:
    def __init__(self, module_name: str):
        """
        The imported module must define:
        - `get_state_class()` function that returns a State class derived from `AbstractState`.
        - `get_all_transitions()` function that returns all possible transitions.
        - `get_transition_schema()` function that returns an integer indicating which transition schema to use.
        - `get_state_name_mapping()` function that returns a mapping of state names for protocol-specific rules.
        """
        self.llm_solution = import_module(module_name)
        self.verifier = None
    
    def set_verifier(self, num_nodes: int, num_rounds: int):
        self.verifier = Z3Verifier(self.llm_solution.get_state_class(), num_nodes, num_rounds)

    def verify_llm_solution(self, counter=True):
        if self.verifier is None:
            raise ValueError("Verifier not set. Call set_verifier() first.")
        self.set_transition_constraints(self.llm_solution.get_transition_schema())
        self.set_pb_rule(self.llm_solution.get_state_name_mapping(), counter)
        self.verifier.verify()

    def set_transition_constraints(self, schema: int = 0):
        match schema:
            case 0:
                self._schema_0()
            case _:
                raise NotImplementedError(f"Schema {schema} not implemented.")

    def _schema_0(self):
        initial_states, transitions = self.llm_solution.get_all_transitions(self.verifier.num_nodes, self.verifier.num_rounds)
        logger.info(f"Total initial transitions: {len(initial_states)}, other transitions: {len(transitions)}")
        transition_c = []
        for r in range(self.verifier.num_rounds):
            for n in range(self.verifier.num_nodes):
                this_node_c = []
                target_transitions = initial_states if r == 0 else transitions
                for t in target_transitions:
                    cur_state_c = [self.verifier.all_states[r][n] == self.verifier.type_mapping[t[0].name]]
                    range_list = [f for f in range(self.verifier.num_nodes) if f != n]
                    msg_c = [
                        self.verifier.all_messages[r][n][f] == self.verifier.type_mapping[t[1][i].name]
                        for i, f in enumerate(range_list)
                    ]
                    next_state_c = z3.Or(self.verifier.all_states[r+1][n] == self.verifier.type_mapping[t[2].name], self.verifier.all_states[r+1][n] == self.verifier.lost_state)
                    this_node_c.append(
                        z3.Implies(
                            z3.And(cur_state_c + msg_c),
                            next_state_c
                        )
                    )
                transition_c.append(z3.And(this_node_c))
        self.verifier.add_constraint(z3.And(transition_c))
    
    def set_pb_rule(self, state_name: dict, counter=True):
        """
        Set Primary Backup protocol specific rules.
        - state_name: {
            "LocalZero": str,
            "LocalOne": str,
            "Zero": str,
            "One": str
          }
        """
        # R1: final states must have corresponding initial states
        r1_c1 = z3.Implies(
            z3.And([self.verifier.all_states[0][n] == self.verifier.type_mapping[state_name['LocalOne']] for n in range(self.verifier.num_nodes)]),
            z3.And([self.verifier.all_states[-1][n] != self.verifier.type_mapping[state_name['Zero']] for n in range(self.verifier.num_nodes)])
        )
        r1_c2 = z3.Implies(
            z3.And([self.verifier.all_states[0][n] == self.verifier.type_mapping[state_name['LocalZero']] for n in range(self.verifier.num_nodes)]),
            z3.And([self.verifier.all_states[-1][n] != self.verifier.type_mapping[state_name['One']] for n in range(self.verifier.num_nodes)])
        )
        # R2: no contradiction in final states
        r2_c = z3.Not(
            z3.And(
                z3.Or([self.verifier.all_states[-1][n] == self.verifier.type_mapping[state_name['One']] for n in range(self.verifier.num_nodes)]),
                z3.Or([self.verifier.all_states[-1][n] == self.verifier.type_mapping[state_name['Zero']] for n in range(self.verifier.num_nodes)])
            )
        )
        if not counter:
            self.verifier.add_constraint(z3.And(r1_c1, r1_c2, r2_c))
        else:
            self.verifier.add_constraint(z3.Not(z3.And(r1_c1, r1_c2, r2_c)))
    
    def fix_states_for_test(self, states_rounds: list[tuple[list, int]]):
        """
        (for test)
        Fix states at specific rounds.

        states_rounds: list of (round_number, [state1, state2, ..., stateN])
        round_number: 0 means initial state before round-0
        """
        for states, round_number in states_rounds:
            self.verifier.add_state_constraints_test(states, round_number)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run_test = False

    # llm_sol = "llms.pb_gemini.v1"
    # llm_sol = "llms.pb_gpt.v1"
    llm_sol = "llms.pb_qwen.v1"
    if run_test:
        import_module(llm_sol).make_test_case()
    else:
        settings = [(5, 4)]
        for num_nodes, num_rounds in settings:
            logging.info(f"Verifying {llm_sol} with {num_nodes} nodes and {num_rounds} rounds.")
            simulator = LLMProtocolSimulator(llm_sol)
            simulator.set_verifier(num_nodes, num_rounds)
            # simulator.fix_states_for_test(import_module(llm_sol).fix_states())
            simulator.verify_llm_solution(counter=True)
