# `itembed` — Item embeddings

This is yet another variation of the well-known word2vec method, proposed by Mikolov et al., applied to unordered sequences, which are commonly referred as itemsets.
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
pip install git+https://github.com/jojolebarjos/itembed.git
```

Please refer to the [documentation](http://jojolebarjos.github.io/itembed) for detailed explanations and examples.


## Changelog

 * 0.6.0 - 2024-xx-xx
    * Migrate to GitHub
    * Rewrite documentation and examples
 * 0.5.0 - 2021-09-03
    * Add weighted itemset support
    * Improve documentation and examples
    * Bug fixes
 * 0.4.2 - 2020-06-10
    * Clean documentation and examples
    * Bug fixes
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
