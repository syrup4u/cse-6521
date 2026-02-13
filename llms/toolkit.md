# Toolkit

## API Definition

One possible version:

| Tool Name | Arguments | Purpose |
| --- | --- | --- |
| `verify_protocol` | `protocol_code`, `protocol_desc` | Runs the full verifier. Returns success or a refining trace feedback (e.g., causal history in `json`). |
| `check_syntax` | `protocol_code` | Fast check. Returns compiler/parser errors. |
| `run_scenario` | `events_list` (e.g., `[send(A, B), crash(A)]`) | Executes a specific sequence. Returns the final state and logs. |
| `visualize_trace` | `trace_object` | Converts a trace into a Mermaid/PlantUML string for visualization. |
| *`check_invariant` | `candidate_invariant` | Checks if a specific formula holds or is inductive. |

### Details

"Me" or "I" refers to LLM.

> About Counterexample

- Trace Minimization: Implement a pass in your verifier that shrinks the counterexample to the shortest possible sequence of events that reproduces the bug.
- State Diffing: Instead of showing the full state at every step, show me only what changed (the diff) between steps $t$ and $t+1$.
- Visual Sequence Diagrams: If your toolkit can render the counterexample into a text-based sequence diagram (like MermaidJS or PlantUML), I can parse the causal flow much faster than raw state dumps.
- Annotated Failures: If possible, identify which specific safety property failed and at what step (e.g., "Invariant `NoTwoLeaders` violated at Step 12").

> About Static Analysis & Syntax Feedback

- Syntax/Type Checking: If I am outputting code (e.g., TLA+ or Python), provide immediate feedback on syntax errors or type mismatches.
- Sanity Checks: Allow me to define simple "unit tests" or assertions. For example, "Does the protocol reach a commit state in a happy path with 0 failures?" If it doesn't do this, there is no point checking for safety yet.

> About Interactive Simulation

- Specific Scenario Execution: Let me call a tool to run a specific deterministic scenario.
  - Example Request: "Run a simulation with 3 nodes where Node 1 crashes after sending Prepare, and tell me the state of Node 2."
- State Inspection: Allow me to query the state of a specific variable or node at a specific time in that trace.

> About Invariant Helper (*)

- Inductive Counterexamples: If I propose an invariant that isn't inductive (it holds initially but isn't preserved by transitions), show me the transition that breaks it.
- Unreachable State Identification: If the verifier finds a bug, but that state is actually unreachable in reality (but reachable in my current weak invariant), tell me: "This state violates safety, but verify if it is actually reachable."

### Possible Workflow

1. LLM draft a protocol.
2. LLM call `check_syntax` (Low cost).
3. LLM call `run_scenario` on a "happy path" to ensure liveness (Medium cost).
4. LLM call `verify_protocol` to check safety (High cost).
5. If verification fails, our toolkit returns a refining trace feedback like a counter-example.
6. LLM analyze the feedback, patch the logic, and repeat.

### Formal Verification

Mainly talking about inductive invariants.

Let's say:

*  $Init$ is your initial state constraints.
*  $Tr(S, S')$ is your transition logic (how state  updates to  based on messages).
*  $Inv(S)$ is the candidate invariant the AI proposes.
*  $Safe(S)$ is your safety requirement (e.g., "no two leaders").

You create a tool `check_inductive_invariant(invariant_code)` that runs these three checks in Z3:

1. **Initiation:** Does the system start in a valid state?
* Z3 Check: `Unsat(Init(S) AND NOT Inv(S))`

2. **Consecution (The Inductive Step):** If we start in a state where the invariant holds, does it *still* hold after one step?
* Z3 Check: `Unsat(Inv(S) AND Tr(S, S') AND NOT Inv(S'))`


3. **Safety:** Does the invariant imply the safety property?
* Z3 Check: `Unsat(Inv(S) AND NOT Safe(S))`

**If all three return UNSAT, the protocol is verified correct.**

Since the LLM (me) is good at guessing logic but bad at rigorous checking, you should build a loop where I "guess" the invariant, and you "check" it.

**New Tool: `verify_invariant`**

* **Input:** A logical formula string (e.g., `forall n1, n2: vote[n1] -> not vote[n2]`).
* **Output (Feedback):**
    * **Pass:** "Invariant Verified!"
    * **Fail - Initiation:** "Failed at start state. Here is the initial state `S`."
    * **Fail - Inductive:** "Failed at transition. Here is a **Counterexample to Induction (CTI)**."
    * **Fail - Safety:** "Invariant holds, but it allows unsafe states."

A **Counterexample to Induction (CTI)** is the most valuable feedback you can give me.

* It is a pair of states $(S, S')$.
* $S$ is a state that *looks* fine (it satisfies my guessed invariant).
* $S'$ is the next state where the invariant breaks.
* **Insight:** This tells me, "Your invariant wasn't strong enough. You allowed a state  that *should* have been impossible."

Invariant Workflow:

1. **I design the protocol** (Transition logic).
2. **I propose a candidate invariant** (The "Fence").
3. **You call** `verify_invariant` using Z3.
4. **If it fails**, you return the **CTI** (The specific transition where the fence broke).
5. **I refine the invariant** (strengthen the fence) or fix the protocol code.
