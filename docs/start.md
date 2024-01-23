# Getting started

...


## Installation

Install from [PyPI](https://pypi.org/project/itembed/):

```
pip install itembed
```

Or install from source, to ensure latest version:

```
pip install git+https://github.com/jojolebarjos/itembed.git
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
    ["apple", "sugar", "flour"],
    ["pear", "sugar", "flour", "butter"],
    ["apple", "pear", "sugar", "buffer", "cinnamon"],
    ["salt", "flour", "oil"],
    # ...
]

labels, indices, offsets = pack_itemsets(itemsets, min_count=2, min_length=2)
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

Third, the `do_batch` method must be invoked multiple times, until convergence.
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
    ["apple", "sugar", "flour"],
    ["pear", "sugar", "flour", "butter"],
    ["apple", "pear", "sugar", "buffer", "cinnamon"],
    ["salt", "flour", "oil"],
    # ...
]

# Pack itemsets into contiguous arrays
labels, indices, offsets = pack_itemsets(itemsets, min_count=2, min_length=2)
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

More examples can be found in `./example/`. See the
[documentation](https://itembed.readthedocs.io/en/stable/) for more detailed
information.


## Performance improvement

As [suggested](https://numba.pydata.org/numba-doc/dev/user/performance-tips.html#intel-svml) in Numba's documentation, Intel's short vector math library can be used to increase performances:

```
conda install -c numba icc_rt
```