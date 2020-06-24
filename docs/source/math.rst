.. _math:


Mathematical Background
=======================

:cite:`wu2017starspace`, :cite:`barkan2016item2vec`...


The Pair Paradygm
-----------------

Item pairs are at the center of :cite:`journals/corr/abs-1301-3781` and its derivatives.
Instead of processing a whole sequence, only two items are considered at a single step. This section discusses how to select them and what they represent.


Input-Output
^^^^^^^^^^^^

The most straightforward way to define an item pair is in the supervised case.
The left-hand side is the input (a.k.a. feature item) and the right-hand side is the output (a.k.a. label item).

...


Skip-Gram
^^^^^^^^^

...


Why Negative Sampling?
----------------------


Softmax Formulation
^^^^^^^^^^^^^^^^^^^

Let :math:`(a, b)` a pair of items, where :math:`a \in A` is the source and :math:`b \in B` the target.
The actual meaning depends on the use case, as
discussed above.

The conditional probability of observing :math:`b` given :math:`a` is defined by a softmax on all possibilities, as it is a regular multi-class task:

.. math::

    P(b \mid a ; \mathbf{u}, \mathbf{v}) = \frac{e^{\mathbf{u}_a^T \mathbf{v}_b}}{\sum_{b'} e^{\mathbf{u}_a^T \mathbf{v}_{b'}}}

The log-likelihood is therefore defined as:

.. math::

    \mathcal{L} (a, b ; \mathbf{u}, \mathbf{v}) = -\log P(b \mid a ; \mathbf{u}, \mathbf{v}) = -\mathbf{u}_a^T \mathbf{v}_b + \log \sum_{b'} e^{\mathbf{u}_a^T \mathbf{v}_{b'}}

.. math::

    \frac{\partial}{\partial \mathbf{u}_a} \mathcal{L} (a, b ; \mathbf{u}, \mathbf{v}) = -\mathbf{v}_b + \sum_{b'} P(b' \mid a ; \mathbf{u}, \mathbf{v}) \mathbf{v}_{b'}

However, this implies a summation over every :math:`b' \in B`, which is computationally expensive for large vocabularies.


Noise Contrastive Estimation Formulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Noise Contrastive Estimation (Gutmann and Hyvärinen :cite:`journals/jmlr/GutmannH10`) is proposed by Mnih and Teh :cite:`conf/icml/MnihT12` as a stable sampling method, to reduce the cost induced by softmax computation.
In a nutshell, the model is trained to distinguish observed (positive) samples from random noise.
Logistic regression is applied to minimize the negative log-likelihood, i.e. cross-entropy of our training example against the :math:`k` noise samples:

.. math::

    \mathcal{L} (a, b) = - \log P(y = 1 \mid a, b) + k \mathbb{E}_{b' \sim Q}\left[ - \log P(y = 0 \mid a, b) \right]

To avoid computating the expectation on the whole vocabulary, a Monte Carlo approximation is applied. :math:`B^* \subseteq B`, with :math:`\vert B^* \vert = k`, is therefore the set of random samples used to estimate it:

.. math::

    \mathcal{L} (a, b) = - \log P(y = 1 \mid a, b) - k \sum_{b' \in B^* \subseteq B} \log P(y = 0 \mid a, b')

We are effectively generating samples from two different distributions: positive pairs are sampled from the empirical training set, while negative pairs come from the noise distribution :math:`Q`.

.. math::

    P(y, b \mid a) = \frac{1}{k + 1} P(b \mid a) + \frac{k}{k + 1} Q(b)

Hence, the probability that a sample came from the training distribution:

.. math::

    P(y = 1 \mid a, b) = \frac{P(b \mid a)}{P(b \mid a) + k Q(b)}

.. math::

    P(y = 0 \mid a, b) = 1 - P(y = 1 \mid a, b)

However, :math:`P(b \mid a)` is still defined as a softmax:

.. math::

    P(b \mid a ; \mathbf{u}, \mathbf{v}) = \frac{e^{\mathbf{u}_a^T \mathbf{v}_b}}{\sum_{b'} e^{\mathbf{u}_a^T \mathbf{v}_{b'}}}

Both Mnih and Teh :cite:`conf/icml/MnihT12` and Vaswani et al. :cite:`conf/emnlp/VaswaniZFC13` arbitrarily set the denominator to 1.
The underlying idea is that, instead of explicitly computing this value, it could be defined as a trainable parameter.
Zoph et al. :cite:`conf/naacl/ZophVMK16` actually report that even when trained, it stays close to 1 with a low variance.

Hence:

.. math::

    P(b \mid a ; \mathbf{u}, \mathbf{v}) = e^{\mathbf{u}_a^T \mathbf{v}_b}

The negative log-likelihood can then be computed as usual:

.. math::

    \mathcal{L} (a, b ; \mathbf{u}, \mathbf{v}) = -\log P (a, b ; \mathbf{u}, \mathbf{v})

Mnih and Teh :cite:`conf/icml/MnihT12` report that using :math:`k = 25` is sufficient to match the performance of the regular softmax.


Negative Sampling Formulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Negative Sampling, popularised by Mikolov et al. :cite:`mikolov2013distributed`, can be seen as an approximation of NCE.
As defined previously, NCE is based on the following:

.. math::

    P(y = 1 \mid a, b ; \mathbf{u}, \mathbf{v}) = \frac{e^{\mathbf{u}_a^T \mathbf{v}_b}}{e^{\mathbf{u}_a^T \mathbf{v}_b} + k Q(b)}

Negative Sampling simplifies this computation by replacing :math:`k Q(b)` by 1.
Note that :math:`k Q(b) = 1` is true when :math:`B^* = B` and :math:`Q` is the uniform distribution.

.. math::

    P(y = 1 \mid a, b ; \mathbf{u}, \mathbf{v}) = \frac{e^{\mathbf{u}_a^T \mathbf{v}_b}}{e^{\mathbf{u}_a^T \mathbf{v}_b} + 1} = \sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right)

Hence:

.. math::

    P(a, b ; \mathbf{u}, \mathbf{v}) = \sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right) \prod_{b' \in B^* \subseteq B} \left( 1 - \sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right) \right)

.. math::

    \mathcal{L} (a, b ; \mathbf{u}, \mathbf{v}) = -\log \sigma \left( \mathbf{u}_a^T \mathbf{v}_b \right) - \sum_{b' \in B^* \subseteq B} \log \left( 1 - \sigma \left( \mathbf{u}_a^T \mathbf{v}_b' \right) \right)

For more details, see Goldberg and Levy's notes :cite:`goldberg2014word2vec`.

To compute the gradient, let us rewrite the loss as:

.. math::

    \mathcal{L} (a, b ; \mathbf{u}, \mathbf{v}) = -\ell_{a, b, 1} - \sum_{b' \in B^* \subseteq B} \ell_{a, b', 0}

where

.. math::

    \ell_{a, b, y} = \log \sigma \left( y - \mathbf{u}_a^T \mathbf{v}_b \right)

Then:

.. math::

    \begin{array}{lll}
    \frac{\partial}{\partial \mathbf{u}_a} \ell (a, b, y) & = & \frac{1}{y - \sigma \left(\mathbf{u}_a^T \mathbf{v}_b \right)}
    \left( - \sigma \left(\mathbf{u}_a^T \mathbf{v}_b \right) \left( 1 - \sigma \left(\mathbf{u}_a^T \mathbf{v}_b \right) \right) \right) \mathbf{v}_b \\
    & = & \left( y - \sigma \left(\mathbf{u}_a^T \mathbf{v}_b \right) \right) \mathbf{v}_b
    \end{array}

And similarly:

.. math::

    \frac{\partial}{\partial \mathbf{v}_b} \ell (a, b, y) = \left( y - \sigma \left(\mathbf{u}_a^T \mathbf{v}_b \right) \right) \mathbf{u}_a


Additional Considerations
-------------------------


Normalization
^^^^^^^^^^^^^

By setting the denominator to 1, as proposed above, the model essentially learns to self-normalize.
However, Devlin et al. :cite:`devlin2014robust` suggest to add a squared error penalty to enforce the equivalence.
Andreas and Klein :cite:`conf/naacl/AndreasK15` even suggest that it should even be sufficient to only normalize a fraction of the training examples and still obtain approximate self-normalising behaviour.


Item distribution balancing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In word2vec, Mikolov et al. :cite:`mikolov2013distributed` use a subsampling approach to reduce the impact of frequent words.
Each word has a probability

.. math::

    P(w_i) = 1 - \sqrt{ \left( \frac{t}{f(w_i)} \right) }

of being discarded, where :math:`f(w_i)` is its frequency and :math:`t` a chosen threshold, typically around :math:`10^{-5}`.


Parallelization
^^^^^^^^^^^^^^^

Hogwild :cite:`conf/nips/RechtRWN11`...


References
----------

.. bibliography:: references.bib
    