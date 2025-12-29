# Thinking Invariance and Generalization in Learning Distributed Protocol

In short, the main idea is to explore:

1. Can we leverage the invariance in distributed protocols to accelerate the training and help the model converge?
2. If we succeed in training a model for a 3-node protocol, can we generalize it to an any-number-of-node protocol?

## Motivation

It is extremely difficult to train a model to learn a distributed protocol, especially some complex ones. Unlike existing protocols, the model is trained based on real numbers instead of some interpretable logical rules, so the size of the input space and intermidiate state matters, which means more nodes, states and rounds will result in a more complicated situation, and the model will be harder to train and converge.

For example, assume there are $N$ nodes and $R$ rounds in a protocol, each round has $N_i$ possible input states and $N_o$ possible output states. Then we have the number of possible inputs to be $R \times N_i^{N}$, while the number of combinations of possible outputs will be $N_o^{R \times N_i^{N}}$. Note that one combination can be seen as a case in one full run of the protocol (reach final round). We can see that even the input space grows exponentially, not to mention the number of possible cases that a model may go through.

Then two ideas come into my mind immediately: at first I was thinking since a 3-node scenario is easy to train on, can we generalize the model to a more nodes scenario? It is kind of similar to the not fully connected NN like CNNs. If we succeed in generalizing the model, we can definitely speed up the learning process because inference is much cheaper and faster than training. Later I remembered there is an implicit property in most of the distributed protocols -- input invariance, i.e., a node does not consider where do the votes come from, it just counts them as the same proportion of the contribution for its decision. This can even be further extended to some weight-based protocols such as weighted voting, Algorand and RepuCoin, since we can "divide" the vote or by simply multiplying the weights. But it does not hold in history-based situations theoretically, where we may need to take the input from last round. Overall, we can use this property to reduce the size of the input space, either by pooling or different representations / embeddings like set rather than the sequence.

And there are two research works that might be helpful for this research project: Set Transformer and Provenance Invariants. The former one fits well with the invariance idea, and the latter one can be supportive to the deductive reasoning of the generalization idea.

## Future Work

Check [TODO](./TODO.md)

## Run the Code

Check [ENV.md](./ENV.md)

## Reference

- [Set Transformer](https://github.com/juho-lee/set_transformer)
- [Provenance Invariants](https://www.usenix.org/conference/osdi25/presentation/zhang-tony)
