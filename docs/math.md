# Mathematical details

The `itembed` framework is built upon word2vec[@mikolov2013efficient], StarSpace[@wu2017starspace], and item2vec[@barkan2017item2vec].
These approaches aim to convert discrete items, such as words or products, into vectors that represent meaningful positions in a high-dimensional space.
This section outlines the mathematical framework behind `itembed`, specifically focusing on the pair prediction mechanism and the concept of negative sampling.


## The pair paradigm

In word2vec, two formulations are proposed for training embeddings:

 * _Skip-Gram Model_:
   This approach predicts the surrounding context for a given word.
   It takes a single input and predict multiple contextual outputs.
 * _Continuous Bag Of Words (CBOW)_:
   Conversely, the CBOW model uses the context of surrounding words to predict a target word, effectively consolidating multiple inputs into a single output.

Despite their distinct methodologies, both the skip-gram and CBOW models rely fundamentally on the concept of pairs.
More precisely, a _feature_ word $a$ is used to predict a _label_ word $b$, representing a single sample pair $(a, b)$.
The models are essentially training a shallow neural network for multi-class classification, on a vast number of these pairs derived from extensive corpora.

The main challenge is to choose training pairs according to the available data.
In unsupervised learning scenarios, the strategy is to select item pairs that occur within the same context, utilizing their co-occurrence to uncover underlying patterns.
For supervised learning tasks, the focus shifts to combining items from separate itemsets, where a pair is composed of a feature item from one itemset and a label item from a different one.

!!! note

    Wu et al.[@wu2017starspace] provide a wide range of applications and examples of pair selection strategies, including recommendation systems and multi-relational graphs.


## Why negative sampling?

Word2vec also addresses the challenge of the computational cost associated with the softmax function used in multi-class classification, especially when dealing with a large number of classes.
Typically, vocabularies consist of tens of thousands of unique words, and because softmax requires normalization across all classes, the computational burden for each update can be significant.
To mitigate this issue, word2vec introduces several simplifications.


### Softmax formulation

Let $(a, b)$ a pair of items, where $a \in A$ is the input and $b \in B$ the output.
The actual meaning depends on the use case, as discussed above.

Furthermore, let $\mathbf{u} \in \mathbb{R}^{|A| \times d}$ and $\mathbf{v} \in \mathbb{R}^{|B| \times d}$ the two embedding sets with dimension size $d$.
We denote $\mathbf{u}_a \in \mathbb{R}^d$ and $\mathbf{v}_b \in \mathbb{R}^d$ the embedding vectors of $a$ and $b$, respectively.
The model outputs a score, akin to similarity, as follows:

$$ f(a, b ; \mathbf{u}, \mathbf{v}) = \mathbf{u}_a^T \mathbf{v}_b $$

In the context of multi-class classification, the [softmax function](https://en.wikipedia.org/wiki/Softmax_function) is used to model the conditional probability of observing class $b$ given input $a$:

$$ P(b \mid a ; \mathbf{u}, \mathbf{v}) = \frac{e^{f(a, b ; \mathbf{u}, \mathbf{v})}}{\sum_{b'} e^{f(a, b ; \mathbf{u}, \mathbf{v})}} $$

The likelihood quantifies how well the model parameters, $\mathbf{u}$ and $\mathbf{v}$, explain the observed data.
During training, these parameters are adjusted to maximize the likelihood, representing the joint probability of the training data.
Assuming independence among input-output pairs, the likelihood is given by:

$$ \mathcal{L} (\mathbf{u}, \mathbf{v}) = \prod_{a,b} P(b \mid a ; \mathbf{u}, \mathbf{v}) $$

For numerical stability and simplification, the Negative Log-Likelihood (NLL) is minimized.
Focusing on a single data pair:

$$ l (a, b ; \mathbf{u}, \mathbf{v}) = -\log P(b \mid a ; \mathbf{u}, \mathbf{v}) $$

In practice, gradient descent is used to minimize $l (a, b ; \mathbf{u}, \mathbf{v})$.
The gradient with respect to $\mathbf{v}_b$ is derived using the chain rule:

$$ \frac{\partial l (a, b ; \mathbf{u}, \mathbf{v})}{\partial \mathbf{v}_b} = \frac{\partial l (a, b ; \mathbf{u}, \mathbf{v})}{\partial P(b \mid a ; \mathbf{u}, \mathbf{v})} \frac{\partial P(b \mid a ; \mathbf{u}, \mathbf{v})}{\partial f(a, b ; \mathbf{u}, \mathbf{v})} \frac{\partial f(a, b ; \mathbf{u}, \mathbf{v})}{\partial \mathbf{v}_b} $$

$$ \frac{\partial l (a, b ; \mathbf{u}, \mathbf{v})}{\partial P(b \mid a ; \mathbf{u}, \mathbf{v})} = -\frac{1}{P(b \mid a ; \mathbf{u}, \mathbf{v})} $$

$$ \frac{\partial P(b \mid a ; \mathbf{u}, \mathbf{v})}{\partial f(a, b ; \mathbf{u}, \mathbf{v})} = P(b \mid a ; \mathbf{u}, \mathbf{v}) \left( 1 - P(b \mid a ; \mathbf{u}, \mathbf{v}) \right) $$

$$ \frac{\partial f(a, b ; \mathbf{u}, \mathbf{v})}{\partial \mathbf{v}_b} = \mathbf{u}_a^T $$

Therefore, and by symmetry:

$$ \frac{\partial l (a, b ; \mathbf{u}, \mathbf{v})}{\partial \mathbf{v}_b} = \left( P(b \mid a ; \mathbf{u}, \mathbf{v}) - 1 \right) \mathbf{u}_a^T $$

$$ \frac{\partial l (a, b ; \mathbf{u}, \mathbf{v})}{\partial \mathbf{u}_a} = \left( P(a \mid b ; \mathbf{u}, \mathbf{v}) - 1 \right) \mathbf{v}_b^T $$

Hence, gradient descent on a softmax activation requires a summation over the whole sets $A$ and $B$, which is computationally expensive for large vocabularies.

!!! note

    A more detailed derivation of the softmax gradient is provided by Miranda[@miranda2017softmax].


### Noise contrastive estimation formulation

Noise Contrastive Estimation (NCE) was introduced by Gutmann and Hyvärinen[@gutmann2010noise] and later proposed by Mnih and Teh[@mnih2012fast] as an efficient sampling method to alleviate the computational burden of the softmax function.
NCE simplifies the task to a binary classification problem, where the model learns to differentiate between actual training data (positive samples) and artificially generated noise (negative samples).

The process involves generating samples from two distinct distributions: positive samples are drawn directly from the empirical training set, whereas negative samples are produced from a predefined noise distribution, denoted by $\eta$.
For every genuine sample pair $(a, b)$, $k$ negative counterparts $(a, b_i')$ are generated, where $b_i'$ are sampled from $\eta$.

For positive samples (where $y=1$), the probability corresponds to the softmax formulation:

$$ P(b \mid y=1, a; \mathbf{u}, \mathbf{v}) = \frac{e^{f(a, b ; \mathbf{u}, \mathbf{v})}}{\sum_{b'} e^{f(a, b ; \mathbf{u}, \mathbf{v})}} $$

For negative samples (where $y=0$), $b$ is sampled from $\eta$.
For instance, assuming $\eta$ is uniform:

$$ P(b \mid y=0) = \frac{1}{|B|} $$

The sampling rates for positive and negative samples lead to the joint probability:

$$ P(y=0) = \frac{k}{k+1}, \qquad P(y=1) = \frac{1}{k+1} $$

$$ P(y, b \mid a; \mathbf{u}, \mathbf{v}) = P(y=0) P(b \mid y=0) + P(y=1) P(b \mid y=1, a; \mathbf{u}, \mathbf{v}) $$

Leveraging conditional probabilities, the binary classification model is defined as following:

$$ P(y = 0 \mid a, b; \mathbf{u}, \mathbf{v}) = \frac{k P(b \mid y=0)}{k P(b \mid y=0) + P(b \mid y=1, a; \mathbf{u}, \mathbf{v})} $$

$$ P(y = 1 \mid a, b; \mathbf{u}, \mathbf{v}) = \frac{P(b \mid y=1, a; \mathbf{u}, \mathbf{v})}{k P(b \mid y=0) + P(b \mid y=1, a; \mathbf{u}, \mathbf{v})} $$

The likelihood for a single input-output pair $(a, b)$ combines the probability of correctly identifying both $b$ as a positive sample and $b_i'$ as negative ones:

$$ l (a, b ; \mathbf{u}, \mathbf{v}) = - \log P(y = 1 \mid a, b; \mathbf{u}, \mathbf{v}) - \sum_{i=1, b_i' \sim \eta}^k \log P(y = 0 \mid a, b_i'; \mathbf{u}, \mathbf{v}) $$

At this point, it should be noted that both terms still depend on the softmax function over the whole $B$ set.
Both Mnih and Teh[@mnih2012fast] and Vaswani et al.[@vaswani2013decoding] discuss the computational cost of the softmax function denominator, acting as a normalization term:

$$ P(b \mid y=1, a; \mathbf{u}, \mathbf{v}) = \frac{e^{f(a, b ; \mathbf{u}, \mathbf{v})}}{\sum_{b'} e^{f(a, b ; \mathbf{u}, \mathbf{v})}} = \frac{e^{f(a, b ; \mathbf{u}, \mathbf{v})}}{Z_{\mathbf{u}, \mathbf{v}}(a)} $$

To mitigate this, one proposed strategy involves introducing trainable parameters $z_a$, aimed at approximating $Z_{\mathbf{u}, \mathbf{v}}(a)$ and being optimized jointly during the training process.
Interestingly, Mnih and Teh found that simply setting $z_a=1$ does not affect model performance.
Supporting this, Zoph et al.[@zoph2016simple] observed that even when $z_a$ is allowed to vary during training, it remains close to 1, exhibiting low variance.
Consequently, a practical approach is to adopt a fixed value of 1 for $z_a$, simplifying the equations and reducing computational overhead without compromising model efficacy:

$$ P(b \mid y=1, a; \mathbf{u}, \mathbf{v}) = e^{f(a, b ; \mathbf{u}, \mathbf{v})} $$

$$ P(y = 1 \mid a, b; \mathbf{u}, \mathbf{v}) = \frac{e^{f(a, b ; \mathbf{u}, \mathbf{v})}}{k P(b \mid y=0) + e^{f(a, b ; \mathbf{u}, \mathbf{v})}} $$

Mnih and Teh [@mnih2012fast] report that using $k = 25$ is sufficient to match the performance of the regular softmax.

!!! note

    By setting the denominator to 1, as proposed above, the model has to essentially learn to self-normalize.
    Devlin et al.[@devlin2014fast] add a squared error penalty to enforce this equivalence.
    Andreas and Klein[@andreas2015and] suggest that it should even be sufficient to only normalize a fraction of the training examples and still obtain approximate self-normalising behaviour.


### Negative sampling formulation

Negative Sampling, popularised by Mikolov et al.[@mikolov2013distributed], can be seen as an approximation of NCE.
It simplifies this computation by replacing $k P(b \mid y=0)$ by 1, leading to the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) $\sigma(x)$.
Note that equality holds, in theory, only when $k = |B|$ and $\eta$ is the uniform distribution.

$$ \sigma(x) = \frac{e^x}{1 + e^x}, \quad \frac{\partial \sigma(x)}{\partial x} = \sigma(x) \left(1 - \sigma(x) \right) $$

$$ P(y = 0 \mid a, b; \mathbf{u}, \mathbf{v}) = \frac{1}{1 + e^{f(a, b ; \mathbf{u}, \mathbf{v})}} = 1 - \sigma\left(f(a, b ; \mathbf{u}, \mathbf{v})\right) $$

$$ P(y = 1 \mid a, b; \mathbf{u}, \mathbf{v}) = \frac{e^{f(a, b ; \mathbf{u}, \mathbf{v})}}{1 + e^{f(a, b ; \mathbf{u}, \mathbf{v})}} = \sigma\left(f(a, b ; \mathbf{u}, \mathbf{v})\right) $$

$$ P(y \mid a, b; \mathbf{u}, \mathbf{v}) = (1-y) P(y = 0 \mid a, b; \mathbf{u}, \mathbf{v}) + y P(y = 1 \mid a, b; \mathbf{u}, \mathbf{v}) $$

Focusing on a single triplet $(a, b, y)$, the resulting NLL and its partial derivative are:

$$ \mathcal{L} (a, b, y; \mathbf{u}, \mathbf{v}) = P(y \mid a, b; \mathbf{u}, \mathbf{v}) $$

$$ l(a, b, y; \mathbf{u}, \mathbf{v}) = - \log \mathcal{L} (a, b, y; \mathbf{u}, \mathbf{v}) $$

$$ \frac{\partial l (a, b ; \mathbf{u}, \mathbf{v})}{\partial f(a, b ; \mathbf{u}, \mathbf{v})} = \sigma(f(a, b ; \mathbf{u}, \mathbf{v})) - y $$

Therefore:

$$ \frac{\partial l (a, b ; \mathbf{u}, \mathbf{v})}{\partial \mathbf{u}_a} = \left( \sigma(\mathbf{u}_a^T \mathbf{v}_b) - y \right) \mathbf{v}_b^T $$

$$ \frac{\partial l (a, b ; \mathbf{u}, \mathbf{v})}{\partial \mathbf{v}_b} = \left( \sigma(\mathbf{u}_a^T \mathbf{v}_b) - y \right) \mathbf{u}_a^T $$

These two formulas are implemented in [`do_step`](api.md#itembed.do_step), as part of the gradient descent algorithm.

!!! note

    For more details about the inner workings of word2vec, see Goldberg and Levy's notes[@goldberg2014word2vec].
