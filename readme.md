
# Itemset Embeddings

Based on [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (a.k.a. word2vec) by Mikolov et al., and its [official implementation](https://github.com/tmikolov/word2vec).

```
pip install git+https://gitlab.com/jojolebarjos/item2vec.git
```

```python
from item2vec import train
import numpy

# Get your own itemsets
itemsets = [
    ['apple', 'sugar', 'flour'],
    ['pear', 'sugar', 'flour', 'butter'],
    # ...
]

# Train using skip-gram
labels, syn0, syn1 = train(itemsets, size=64)

# Both embedding sets are usable, just choose one
syn = syn0

# Normalize for comparison
n_syn = syn / numpy.sqrt((syn ** 2).sum(axis=1))[:, None]

# Compute cosine distances to all other labels
ref = labels.index('apple')
distances = n_syn @ n_syn[ref]

# Show closest neighbors
print(f'#{ref} {labels[ref]}:')
for i in numpy.argsort(-distances)[:10]:
    print(f'  #{i} {labels[i]} ({distances[i]:0.4f})')
```
