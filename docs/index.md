# Welcome to `itembed`

This is yet another variation of the well-known _word2vec_ method, proposed by Mikolov et al.[@mikolov2013efficient], applied to unordered sequences, which are commonly referred as itemsets. The contribution of _itembed_ is twofold:

 1. Modifying the base algorithm to handle unordered sequences, which has an
    impact on the definition of context windows;
 2. Using the two embedding sets introduced in _word2vec_ for supervised
    learning.

A similar philosophy is described by Wu et al. in _StarSpace_[@wu2017starspace] and by Barkan and Koenigstein in _item2vec_[@barkan2017item2vec].
_itembed_ uses Numba[@lam2015numba] to achieve high performances.


## Citation

If you use this software in your work, it would be appreciated if you would cite this tool, for instance using the following Bibtex reference:

```bibtex
@software{itembed,
  author = {Johan Berdat},
  title = {itembed},
  url = {https://gitlab.com/jojolebarjos/itembed},
  version = {0.5.0},
  date = {2021-09-03},
}
```
