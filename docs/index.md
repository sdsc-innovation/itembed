# Welcome to `itembed`

This is yet another variation of the well-known word2vec method, proposed by Mikolov et al.[@mikolov2013efficient], applied to unordered sequences, which are commonly referred as itemsets.
The contribution of `itembed` is twofold:

 1. Modifying the base algorithm to handle unordered sequences, which has an impact on the definition of context windows;
 2. Using the two embedding sets introduced in word2vec for supervised learning.

A similar philosophy is described by Wu et al. in StarSpace[@wu2017starspace] and by Barkan and Koenigstein in item2vec[@barkan2017item2vec].
`itembed` uses Numba[@lam2015numba] to achieve high performances.


## Citation

If you use this software in your work, it would be appreciated if you would cite this tool, for instance using the following Bibtex reference:

```bibtex
@software{itembed,
  author = {Johan Berdat},
  title = {itembed},
  url = {https://github.com/jojolebarjos/itembed},
  version = {0.6.0},
  date = {2024-xx-xx},
}
```
