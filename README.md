# `itembed` — Item embeddings

This is yet another variation of the well-known word2vec method, proposed by Mikolov et al., applied to unordered sequences, which are commonly referred to as itemsets.
The contribution of `itembed` is twofold:

 1. Modifying the base algorithm to handle unordered sequences, which has an impact on the definition of context windows;
 2. Using the two embedding sets introduced in word2vec for supervised learning.

A similar philosophy is described by Wu et al. in StarSpace and by Barkan and Koenigstein in item2vec.
`itembed` uses Numba to achieve high performances.


## Getting started

Install from [PyPI](https://pypi.org/project/itembed/):

```
pip install itembed
```

Or install from source, to ensure latest version:

```
pip install git+https://github.com/sdsc-innovation/itembed.git
```

Please refer to the [documentation](http://sdsc-innovation.github.io/itembed) for detailed explanations and examples.
