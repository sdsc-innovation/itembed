
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


## Relevant links

 * Hierarchy-related:
    * [Joint Learning of Hierarchical Word Embeddings from a Corpus and a Taxonomy](https://openreview.net/forum?id=S1xf-W5paX)
    * [Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures](https://arxiv.org/abs/1805.06627)
    * [Improved Representation Learning for Predicting Commonsense Ontologies](https://arxiv.org/pdf/1708.00549.pdf)
    * [ORDER-EMBEDDINGS OF IMAGES AND LANGUAGE](https://arxiv.org/pdf/1511.06361.pdf)
    * [HIERARCHICAL DENSITY ORDER EMBEDDINGS](https://arxiv.org/pdf/1804.09843.pdf)
    * [Entity Hierarchy Embedding](http://www.cs.cmu.edu/~poyaoh/data/acl15entity.pdf)
