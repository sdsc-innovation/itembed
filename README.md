
# Itemset embeddings

This is yet another variation of the well-known _word2vec_ method, proposed by
[Mikolov et al.](#ref_word2vec), applied to unordered sequences, which are
commonly referred as itemsets. The contribution of _itembed_ is twofold:

 1. Modifying the base algorithm to handle unordered sequences, which has an
    impact on the definition of context windows;
 2. Using the two embedding sets introduced in _word2vec_ for supervised
    learning.

A similar philosophy is described by [Wu et al.](#ref_starspace) in
_StarSpace_ and by [Barkan and Koenigstein](#ref_item2vec) in _item2vec_.
_itembed_ uses [Numba](#ref_numba) to achieve high performances.


## Installation

Install from [PyPI](https://pypi.org/project/itembed/):

```
pip install itembed
```

Or install from source, to ensure latest version:

```
pip install git+https://gitlab.com/jojolebarjos/itembed.git
```


## Getting started

Itemsets must be provided as so-called packed arrays, i.e. a pair of integer
arrays describing _indices_ and _offsets_. The index array is defined as the
concatenation of all N itemsets. The offset array contains the N+1 boundaries.

```python
import numpy as np

indices = np.array([
    0, 1, 4, 7,
    0, 1, 6,
    2, 3, 5, 6, 7,
], dtype=np.int32)

offsets = np.array([
    0, 4, 7, 12
])
```

This is similar to [compressed sparse matrices](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html):

```python
from scipy.sparse import csr_matrix

dense = np.array([
    [1, 1, 0, 0, 1, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 1, 1, 1],
], dtype=np.int32)

sparse = csr_matrix(dense)

assert (indices == sparse.indices).all()
assert (offsets == sparse.indptr).all()

```

Training methods do not handle other data types. Also note that:

 * indices start at 0;
 * item order in an itemset is not important;
 * an itemset can contain duplicated items;
 * itemsets order is not important;
 * there is no weight associated to items, nor itemsets.

However, a small helper is provided for simple cases:

```python
from itembed import pack_itemsets

itemsets = [
    ['apple', 'sugar', 'flour'],
    ['pear', 'sugar', 'flour', 'butter'],
    ['apple', 'pear', 'sugar', 'buffer', 'cinnamon'],
    ['salt', 'flour', 'oil'],
    # ...
]

labels, indices, offsets = pack_itemsets(itemsets, min_count=2)
num_label = len(labels)
```

The next step is to define at least one task. For now, let us stick to the
unsupervised case, where co-occurrence is used as knowledge source. This is
similar to the continuous bag-of-word and continuous skip-gram tasks defined
in _word2vec_.

First, two embedding sets must be allocated. Both should capture the same
information, and one is the complement of the other. This is a not-so
documented question of _word2vec_, but empirical results have shown that it is
better than reusing the same set twice.

```python
from itembed import initialize_syn

num_dimension = 64
syn0 = initialize_syn(num_label, num_dimension)
syn1 = initialize_syn(num_label, num_dimension)
```

Second, define a task object that holds all the descriptors:

```python
from itembed import UnsupervisedTask

task = UnsupervisedTask(indices, offsets, syn0, syn1, num_negative=5)
```

Third, the `do_batch`method must be invoked multiple times, until convergence.
Another helper is provided to handle the training loop. Note that, due to a
different sampling strategy, a larger number of iteration is needed.

```python
from itembed import train

train(task, num_epoch=100)
```

The full code is therefore as follows:

```python
import numpy as np

from itembed import (
    pack_itemsets,
    initialize_syn,
    UnsupervisedTask,
    train,
)

# Get your own itemsets
itemsets = [
    ['apple', 'sugar', 'flour'],
    ['pear', 'sugar', 'flour', 'butter'],
    ['apple', 'pear', 'sugar', 'buffer', 'cinnamon'],
    ['salt', 'flour', 'oil'],
    # ...
]

# Pack itemsets into contiguous arrays
labels, indices, offsets = pack_itemsets(itemsets, min_count=2)
num_label = len(labels)

# Initialize embeddings sets from uniform distribution
num_dimension = 64
syn0 = initialize_syn(num_label, num_dimension)
syn1 = initialize_syn(num_label, num_dimension)

# Define unsupervised task, i.e. using co-occurrences
task = UnsupervisedTask(indices, offsets, syn0, syn1, num_negative=5)

# Do training
# Note: due to a different sampling strategy, more epochs than word2vec are needed
train(task, num_epoch=100)

# Both embedding sets are equivalent, just choose one of them
syn = syn0
```

More examples can be found in `./example/`.


## Performance improvement

As [suggested](https://numba.pydata.org/numba-doc/dev/user/performance-tips.html#intel-svml) in Numba's documentation, Intel's short vector math library can be used to increase performances:

```
conda install -c numba icc_rt
```


## References

<ol>
    <li id="ref_word2vec">
        <i>Efficient Estimation of Word Representations in Vector Space</i>,
        2013,
        Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean,
        https://arxiv.org/abs/1301.3781
    </li>
    <li id="ref_starspace">
        <i>StarSpace: Embed All The Things!</i>,
        2017,
        Ledell Wu, Adam Fisch, Sumit Chopra, Keith Adams, Antoine Bordes, Jason Weston,
        https://arxiv.org/abs/1709.03856
    </li>
    <li id="ref_item2vec">
        <i>Item2Vec: Neural Item Embedding for Collaborative Filtering</i>,
        2016,
        Oren Barkan, Noam Koenigstein,
        https://arxiv.org/abs/1603.04259
    </li>
    <li id="ref_numba">
        <i>Numba: a LLVM-based Python JIT compiler</i>,
        2015,
        Siu Kwan Lam, Antoine Pitrou, Stanley Seibert,
        https://doi.org/10.1145/2833157.2833162
    </li>
</ol>


## Changelog

 * 0.4.1 - 2020-05-13
    * Clean and rename, to avoid confusion
 * 0.4.0 - 2020-05-04
    * Refactor to make training task explicit
    * Add supervised task
 * 0.3.0 - 2020-03-26
    * Complete refactor to increase performances and reusability
 * 0.2.1 - 2020-03-24
    * Allow keyboard interruption
    * Fix label count argument
    * Fix learning rate issue
    * Add optimization flags to Numba JIT
 * 0.2.0 - 2019-11-08
    * Clean and refactor
    * Allow training from plain arrays
 * 0.1.0 - 2019-09-13
    * Initial version
