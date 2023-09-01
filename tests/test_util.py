import numpy as np
from numpy.testing import assert_equal

from itembed import pack_itemsets


def test_pack_itemsets():

    itemsets = [
        ["apple"],
        ["apple", "sugar", "flour"],
        ["pear", "sugar", "flour", "butter"],
        ["apple", "pear", "sugar", "butter", "cinnamon"],
        ["salt", "flour", "oil"],
    ]

    labels, indices, offsets = pack_itemsets(itemsets, min_length=2)

    assert labels == [
        "apple",
        "sugar",
        "flour",
        "pear",
        "butter",
        "cinnamon",
        "salt",
        "oil",
    ]
    assert_equal(indices, [0, 1, 2, 3, 1, 2, 4, 0, 3, 1, 4, 5, 6, 2, 7])
    assert_equal(offsets, [0, 3, 7, 12, 15])
