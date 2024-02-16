# Getting started

In `itembed`, we handle discrete entities called _items_.
These items are part of larger groups or _domains_, classified by their common features.
Our primary interaction with these items occurs within _itemsets_, which are unordered collections that are observed in the data.

Consider the example of cooking recipes: essentially, these are unordered compilations of ingredients, each representing a piece of the broader catalogue of available items.
From a batch of chocolate cookies to a classic cheesecake, each recipe is an itemset within the ingredient domain.
Notably, an ingredient may appear across various itemsets, and sometimes, it might even be listed multiple times within the same recipe.

```py
chocolate_cookies = [
    "butter",
    "white sugar",
    "brown sugar",
    "egg",
    "vanilla extract",
    "all-purpose flour",
    "baking soda",
    "salt",
    "semi-sweet chocolate"
]

cheesecake = [
    "graham cracker",
    "butter",
    "cream cheese",
    "white sugar",
    "sour cream",
    "egg",
    "vanilla extract",
    "lemon zest"
]
```

Just as ingredients are combined to create your favorite dish, words blend together to create sentences.
This analogy helps us highlight a fundamental concept: things, whether they are words or ingredients, get their meaning from the contexts in which they are placed.
This principle is key in word2vec[@mikolov2013efficient], which deduces the significance of words based on their surroundings.
It is a bit like noticing that "apple", "pear", and "blueberry" often end up in pie recipes, hinting at the concept of fruits.
The algorithm picks up on these groupings, bringing similar words closer in the semantic space, helping us understand how words fit together to convey ideas.

`itembed` is essentially bringing this approach to unordered sequence of items.
The main difference lies in how it treats context size, given that word2vec operates with a sliding window over sentences.
Nevertheless, the outcome is the same: to derive a numerical representation for each item in a domain, encapsulated as a dense vector of floating-point numbers.
This process is commonly known as generating _embeddings_ or creating a latent space representation.


## Data representation

At its lowest level, `itembed` is manipulating NumPy arrays.
It requires that itemsets be formatted as _packed arrays_, comprising two one-dimensional integer arrays that denote _indices_ and _offsets_
The indices array is defined as the concatenation of all $N$ itemsets, while the offsets array delineates the $N+1$ demarcations.

Accordingly, our cooking recipes are represented as follows:

```py
import numpy as np

labels = [
    "butter",
    "white sugar",
    "egg",
    "vanilla extract",
    "brown sugar",
    "all-purpose flour",
    "baking soda",
    "salt",
    "semi-sweet chocolate",
    "graham cracker",
    "cream cheese",
    "sour cream",
    "lemon zest",
]

indices = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    0, 1, 2, 3, 10, 11, 12,
], dtype=np.int32)

offsets = np.array([
    0, 9, 17,
])
```

This is similar to [SciPy](https://scipy.org/)'s [compressed sparse arrays](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html):

```py
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array

dense = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
], dtype=np.int32)

sparse = csr_array(dense)

assert_array_equal(indices, sparse.indices)
assert_array_equal(offsets, sparse.indptr)
```

For convenience, higher-level helpers are provided, such as [`pack_itemsets`](api.md#itembed.pack_itemsets):

```python
from itembed import pack_itemsets

itemsets = [
    chocolate_cookies,
    cheesecake,
]

labels, indices, offsets = pack_itemsets(itemsets)
```

At this point, we have encoded two itemsets into a packed array.
The ingredient domain is represented as a sequence of labels, which are referenced by our indices.


## Task definition

At the core of the training procedure are _tasks_, which define the input data to work with, as well as some hyperparameters.
Tasks fall into two categories:

 1. _Unsupervised tasks_ learn from the co-occurrence of items within the same itemsets (i.e. sharing the same domain), as introduced by word2vec;
 2. _Supervised tasks_ capture explicit relationships between items across different itemsets and domains, as proposed by StarSpace[@wu2017starspace].

Although our current example focuses solely on a single domain — ingredients — more complex scenarios might encompass multiple domains.
This broader application is demonstrated in the detailed [cooking recipes](notebooks/recipes.ipynb) example, which integrates ingredients and cuisine styles into a unified training framework through two distinct tasks.

In addition to itemsets, a task also requires the embedding sets, or _synsets_, which are adjusted during the training phase.
These synsets constitute the end product of the training process, typically starting from a state of random initialization.
Over the course of training, they are iteratively moved within this multidimensional space based on the interactions observed within the tasks.

An unsupervised task, for example, requires a pair of synsets:

```py
from itembed import initialize_syn, UnsupervisedTask

num_label = len(labels)
num_dimension = 64

syn0 = initialize_syn(num_label, num_dimension)
syn1 = initialize_syn(num_label, num_dimension)

task = UnsupervisedTask(indices, offsets, syn0, syn1, num_negative=5)
```

!!! note

    The need for two synsets rather than a single one is discussed in the [mathematical details](math.md).
    This section also delves into the rationale behind the selection of the number of negative samples.


## Training loop

The training approach in `itembed` is straightforward.
The principal objective for each task is to refine the embeddings so they accurately reflect the patterns within the observed itemsets, using a classic approach of gradient descent with mini-batches.
In every step of the process, tasks produce a small set of samples.
These samples are then employed to modify the embeddings, following a specific learning rate.

In this framework, each task is provided with its own batch iterator.
This is particularly useful when dealing with multiple tasks that vary in sample size; to ensure an equitable contribution from each task, those with fewer samples are cycled through multiple times.
Additionally, the influence of each task can be calibrated using a weighting factor.

```py
learning_rate = 0.01

for step in range(len(task)):
    task.do_batch(learning_rate)
```

An important component of the training schedule is the gradual reduction of the learning rate over time.
The [`train`](api.md#itembed.train) function streamlines this aspect for us:

```py
from itembed import train

train(task, num_epoch=100)
```


## Complete example

Our embeddings have been successfully trained.
In the context of unsupervised tasks, a single set of embeddings is sufficient as they represent complementary latent spaces.
These embeddings are now ready for immediate use or can be stored for future application.

```py
import numpy as np

from itembed import (
    pack_itemsets,
    initialize_syn,
    UnsupervisedTask,
    train,
)

# Get a dataset of itemsets
itemsets = [
    [
        "butter",
        "white sugar",
        "brown sugar",
        "egg",
        "vanilla extract",
        "all-purpose flour",
        "baking soda",
        "salt",
        "semi-sweet chocolate"
    ],
    [
        "graham cracker",
        "butter",
        "cream cheese",
        "white sugar",
        "sour cream",
        "egg",
        "vanilla extract",
        "lemon zest"
    ],
    # ... ideally many more itemsets
]

# Encode them as a packed array
labels, indices, offsets = pack_itemsets(itemsets)

# We are training an embedding vector for each unique item
num_label = len(labels)

# We arbitrarily choose the size of the latent space
num_dimension = 64

# Two embedding sets are initialized for our unsupervised task
syn0 = initialize_syn(num_label, num_dimension)
syn1 = initialize_syn(num_label, num_dimension)
task = UnsupervisedTask(indices, offsets, syn0, syn1, num_negative=5)

# Use gradient descent to fit the training set
train(task, num_epoch=100)

# Both embedding sets are equivalent, just choose one of them
syn = syn0

# Show the embedding of a specific item
index = labels.index("white sugar")
print(syn[index])
```
