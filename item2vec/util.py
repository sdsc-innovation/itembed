
from collections import Counter

import numpy as np

from tqdm import tqdm

from .optimization import do_unsupervised_steps


def pack_itemsets(itemsets, min_count=1, min_length=2):
    """Convert itemset collection to packed indices.

    Args:
        itemsets (list of list of object): list of sets of hashable objects.
        min_count (int): minimal frequency count to be kept (default: 1).
        min_length (int): minimal itemset length (default: 2).

    Returns:
        labels (list of object): mapping from indices to labels.
        indices (int32, num_item): packed index array.
        offsets (int32, num_itemset + 1): itemsets offsets in packed array.

    """

    # Count labels
    counter = Counter()
    for itemset in itemsets:
        counter.update(itemset)
    if None in counter:
        del counter[None]

    # Define label list
    labels = [l for l, c in counter.most_common() if c >= min_count]
    label_map = {l : i for i, l in enumerate(labels)}

    # Generate indices
    indices = []
    offsets = [0]
    for itemset in itemsets:
        itemset_indices = []
        for label in itemset:
            index = label_map.get(label)
            if index is not None:
                itemset_indices.append(index)
        if len(itemset_indices) >= min_length:
            indices.extend(itemset_indices)
            offsets.append(len(indices))

    # Convert to arrays
    indices = np.array(indices, dtype=np.int32)
    offsets = np.array(offsets, dtype=np.int32)
    return labels, indices, offsets


def initialize_syn(num_label, num_dimension, method='uniform'):
    """Allocate and initialize embedding set.

    Args:
        num_label (int32): number of labels.
        num_dimension (int32): size of embeddings.

    Returns:
        syn (float32, num_label x num_dimension): embedding set.

    """

    if method == 'zero':
        syn = np.zeros((num_label, num_dimension), dtype=np.float32)
    elif method == 'uniform':
        syn = np.random.rand(num_label, num_dimension).astype(np.float32)
        syn -= 0.5
        syn /= num_dimension
    else:
        raise KeyError(method)
    return syn


def train(
    itemsets,
    min_count=1,
    min_length=2,
    num_dimension=100,
    num_epoch=5,
    num_negative=5,
    starting_alpha=0.025,
):
    """Train embeddings from collections.

    Example:
        Toy dataset, where items are strings:

        >>> itemsets = [
        ...     ['apple', 'sugar', 'flour'],
        ...     ['pear', 'sugar', 'flour', 'butter'],
        ...     ['pear', 'apple', 'lemon']
        ... ]
        >>> labels, syn0, syn1 = train(itemsets, num_dimension=16)

    """

    labels, indices, offsets = pack_itemsets(itemsets, min_count, min_length)
    num_label = len(labels)
    syn0 = initialize_syn(num_label, num_dimension)
    syn1 = initialize_syn(num_label, num_dimension, method='zero')
    _train(indices, offsets, syn0, syn1, num_epoch, num_negative, starting_alpha)
    return labels, syn0, syn1


def _train(indices, offsets, syn0, syn1, num_epoch, num_negative, starting_alpha):
    num_label, num_dimension = syn0.shape
    num_itemset = offsets.shape[0] - 1
    tmp_syn = np.empty(num_dimension, dtype=np.float32)
    try:

        step = 0
        step_count = num_epoch * indices.shape[0]
        with tqdm(total=step_count) as progress:
            for epoch in range(num_epoch):

                # For each context
                for i in range(num_itemset):
                    itemset = indices[offsets[i]:offsets[i + 1]]
                    length = itemset.shape[0]
                    if length >= 2:

                        # Update learning rate
                        alpha = (1 - step / step_count) * starting_alpha

                        # Apply optimized step
                        do_unsupervised_steps(itemset, syn0, syn1, tmp_syn, num_negative, alpha)

                    # Move to next context
                    step += length
                    progress.update(length)

    except KeyboardInterrupt:
        pass
