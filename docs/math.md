# Mathematical details

The `itembed` framework is built upon word2vec[@mikolov2013efficient], StarSpace[@wu2017starspace], and item2vec[@barkan2017item2vec].
These methodologies collectively aim to convert discrete items, such as words or products, into vectors that represent meaningful positions in a high-dimensional space.
This section outlines the mathematical framework behind `itembed`, specifically focusing on the pair prediction mechanism and the concept of negative sampling.


## The pair paradigm

In word2vec, two approaches are depicted for training embeddings:

 * _Skip-Gram Model_:
   This approach predicts the surrounding context for a given target word.
   It is characterized by its ability to take a single input and predict multiple contextual outputs.
 * _Continuous Bag Of Words (CBOW)_:
   Conversely, the CBOW model uses the context of surrounding words to predict a target word, effectively consolidating multiple inputs into a single output.

Despite their distinct methodologies, both the skip-gram and CBOW models rely fundamentally on the concept of item pairs.
This could involve a word and its surrounding context in the skip-gram model or the context pointing towards a word in the CBOW model.
The models are essentially training a shallow neural network for multi-class classification, on a vast number of these pairs derived from extensive corpora.

In unsupervised learning scenarios within `itembed`, the strategy is to select item pairs that occur within the same context, utilizing their co-occurrence to uncover underlying patterns.
For supervised learning tasks, the focus shifts to combining items from separate itemsets, where a pair is composed of a 'feature item' from one itemset and a 'label item' from a different one.


## Why negative sampling?

Word2vec also addresses the challenge of the computational cost associated with the softmax function used in multi-class classification, especially when dealing with a large number of classes.
Typically, vocabularies consist of tens of thousands of unique words, and because softmax necessitates normalization across all classes, the computational burden for each update can be significant.
To mitigate this issue, word2vec introduces several simplification techniques.


### Softmax formulation

Let $(a, b)$ a pair of items, where $a \in A$ is the source and $b \in B$ the target.
The actual meaning depends on the use case, as discussed above.

Furthermore, let $\mathbf{u} \in \mathbb{R}^{|A| \times d}$ and $\mathbf{v} \in \mathbb{R}^{|B| \times d}$ the two embedding sets with dimension size $d$.
We denote $\mathbf{u}_a \in \mathbb{R}^d$ and $\mathbf{v}_b \in \mathbb{R}^d$ the embedding vectors of $a$ and $b$, respectively.

The conditional probability of observing $b$ given $a$ is defined by a softmax on all possibilities:

$$ P(b \mid a ; \mathbf{u}, \mathbf{v}) = \frac{e^{\mathbf{u}_a^T \mathbf{v}_b}}{\sum_{b'} e^{\mathbf{u}_a^T \mathbf{v}_{b'}}} $$

The negative log-likelihood is therefore defined as:

$$ \mathcal{L} (a, b ; \mathbf{u}, \mathbf{v}) = -\log P(b \mid a ; \mathbf{u}, \mathbf{v}) = -\mathbf{u}_a^T \mathbf{v}_b + \log \sum_{b'} e^{\mathbf{u}_a^T \mathbf{v}_{b'}} $$

In practice, gradient descent is used to minimize \mathcal{L}.
The update rule for $\mathbf{u}_a$ is based on its partial derivative:

$$ \frac{\partial}{\partial \mathbf{u}_a} \mathcal{L} (a, b ; \mathbf{u}, \mathbf{v}) = -\mathbf{v}_b + \sum_{b'} P(b' \mid a ; \mathbf{u}, \mathbf{v}) \mathbf{v}_{b'} $$

However, this implies a summation over every $b \in B$, which is computationally expensive for large vocabularies.


### Noise contrastive estimation formulation

Noise Contrastive Estimation (NCE), introduced by Gutmann and Hyvärinen [@gutmann2010noise], is proposed by Mnih and Teh [@mnih2012fast] as a stable sampling method, to reduce the cost induced by softmax computation.
In a nutshell, the model is trained to distinguish observed (positive) samples from random noise.
Logistic regression is applied to minimize the negative log-likelihood, i.e. cross-entropy of our training example against the $k$ noise samples:

$$ \mathcal{L} (a, b) = - \log P(y = 1 \mid a, b) + k \mathbb{E}_{b' \sim Q}\left[ - \log P(y = 0 \mid a, b) \right] $$

To avoid computating the expectation on the whole vocabulary, a Monte Carlo approximation is applied. $B^* \subseteq B$, with $\vert B^* \vert = k$, is therefore the set of random samples used to estimate it:

$$ \mathcal{L} (a, b) = - \log P(y = 1 \mid a, b) - k \sum_{b' \in B^* \subseteq B} \log P(y = 0 \mid a, b') $$

We are effectively generating samples from two different distributions: positive pairs are sampled from the empirical training set, while negative pairs come from the noise distribution $Q$.

$$ P(y, b \mid a) = \frac{1}{k + 1} P(b \mid a) + \frac{k}{k + 1} Q(b) $$

Hence, the probability that a sample came from the training distribution:

$$ P(y = 1 \mid a, b) = \frac{P(b \mid a)}{P(b \mid a) + k Q(b)} $$

$$ P(y = 0 \mid a, b) = 1 - P(y = 1 \mid a, b) $$

However, $P(b \mid a)$ is still defined as a softmax:

$$ P(b \mid a ; \mathbf{u}, \mathbf{v}) = \frac{e^{\mathbf{u}_a^T \mathbf{v}_b}}{\sum_{b'} e^{\mathbf{u}_a^T \mathbf{v}_{b'}}} $$

Both Mnih and Teh [@mnih2012fast] and Vaswani et al. [@vaswani2013decoding] arbitrarily set the denominator to 1.
The underlying idea is that, instead of explicitly computing this value, it could be defined as a trainable parameter.
Zoph et al. [@zoph2016simple] actually report that even when trained, it stays close to 1 with a low variance.

Hence:

$$ P(b \mid a ; \mathbf{u}, \mathbf{v}) = e^{\mathbf{u}_a^T \mathbf{v}_b} $$

The negative log-likelihood can then be computed as usual:

$$ \mathcal{L} (a, b ; \mathbf{u}, \mathbf{v}) = -\log P (a, b ; \mathbf{u}, \mathbf{v}) $$

Mnih and Teh [@mnih2012fast] report that using $k = 25$ is sufficient to match the performance of the regular softmax.


### Negative sampling formulation

Negative Sampling, popularised by Mikolov et al. [@mikolov2013distributed], can be seen as an approximation of NCE.
As defined previously, NCE is based on the following:

$$ P(y = 1 \mid a, b ; \mathbf{u}, \mathbf{v}) = \frac{e^{\mathbf{u}_a^T \mathbf{v}_b}}{e^{\mathbf{u}_a^T \mathbf{v}_b} + k Q(b)} $$

Negative Sampling simplifies this computation by replacing $k Q(b)$ by 1.
Note that $k Q(b) = 1$ is true when $B^* = B$ and $Q$ is the uniform distribution.

$$ P(y = 1 \mid a, b ; \mathbf{u}, \mathbf{v}) = \frac{e^{\mathbf{u}_a^T \mathbf{v}_b}}{e^{\mathbf{u}_a^T \mathbf{v}_b} + 1} = \sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right) $$

Hence:

$$ P(a, b ; \mathbf{u}, \mathbf{v}) = \sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right) \prod_{b' \in B^* \subseteq B} \left( 1 - \sigma \left( \mathbf{u}_a^T \mathbf{v}_{b'} \right) \right) $$

$$ \mathcal{L} (a, b ; \mathbf{u}, \mathbf{v}) = -\log \sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right) - \sum_{b' \in B^* \subseteq B} \log \left( 1 - \sigma \left( \mathbf{u}_a^T \mathbf{v}_{b'} \right) \right) $$

For more details, see Goldberg and Levy's notes [@goldberg2014word2vec].


### Gradient computation

In order to apply gradient descent, partial derivatives must be computed.
As this is a sum, let us identify the two main terms:

$$
    \begin{array}{lll}
    \frac{\partial}{\partial \mathbf{u}_a} -\log \sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right) & = &
    -\frac{\sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right) \left( 1 - \sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right) \right) }{\sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right)} \mathbf{v}_b \\
    & = & \left( \sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right) - 1 \right) \mathbf{v}_b
    \end{array}
$$

$$
    \begin{array}{lll}
    \frac{\partial}{\partial \mathbf{u}_a} -\log \left( 1 - \sigma \left( \mathbf{u}_a^T \mathbf{v}_{b'} \right) \right) & = &
    -\frac{- \sigma \left( \mathbf{u}_a^T \mathbf{v}_{b'} \right) \left( 1 - \sigma \left( \mathbf{u}_a^T \mathbf{v}_{b'} \right) \right) }{1 - \sigma \left( \mathbf{u}_a^T \mathbf{v}_{b'} \right)} \mathbf{v}_{b'} \\
    & = & \sigma \left( \mathbf{u}_a^T \mathbf{v}_{b'} \right) \mathbf{v}_{b'}
    \end{array}
$$

As both terms are similar, we can rewrite them using the associated label $y$:

$$ \ell_{a, b, y} = \left( \sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right) - y \right) \mathbf{v}_b $$

Therefore, the overall gradient is:

$$
    \frac{\partial}{\partial \mathbf{u}_a} \mathcal{L} (a, b ; \mathbf{u}, \mathbf{v}) =
    \ell_{a, b, 1} + \sum_{b' \in B^* \subseteq B} \ell_{a, b', 0}
$$

A similar expansion can be done for $\frac{\partial}{\partial \mathbf{v}_b} \mathcal{L} (a, b ; \mathbf{u}, \mathbf{v})$.


## Additional considerations


### Normalization

By setting the denominator to 1, as proposed above, the model essentially learns to self-normalize.
However, Devlin et al. [@devlin2014fast] suggest to add a squared error penalty to enforce the equivalence.
Andreas and Klein [@andreas2015and] even suggest that it should even be sufficient to only normalize a fraction of the training examples and still obtain approximate self-normalising behaviour.


### Item distribution balancing

In word2vec, Mikolov et al. [@mikolov2013distributed] use a subsampling approach to reduce the impact of frequent words.
Each word has a probability

$$ P(w_i) = 1 - \sqrt{ \left( \frac{t}{f(w_i)} \right) } $$

of being discarded, where $f(w_i)$ is its frequency and $t$ a chosen threshold, typically $10^{-5}$.
