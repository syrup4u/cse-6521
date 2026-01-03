# Todo List

- Decouple main process (normal train/evaluation, generalization)
- Z3
- Generalization Architecture (has some problems, cannot be fully isolated group, may try exchange at the last round)
- maintain `__init__.py`
- Evaluate: multi-process to accelerate the evaluation in large dataset.
- Baysian Optimizer

## Potential Optimization

1. encode the initial states (first round messages) into the input.
2. smooth sampling -- by history failed cases with count as the importance factor.
3. extended validation (use more-nodes / more-rounds cases, not as the training dataset but validation dataset).
4. reward redesign.
5. Input 'Generator' (7 nodes 3 rounds == 49,717,376 input patterns, each patterns is about 200 bytes, causes 10G+ memory cost)
