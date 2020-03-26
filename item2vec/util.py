
from collections import Counter

import numpy as np

from tqdm import tqdm

from .optimization import do_unsupervised_batch


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
    initial_learning_rate=0.025,
):
    """Train embeddings from collections.

    Args:
        itemsets (list of list of object): list of sets of hashable objects.
        min_count (int): minimal frequency count to be kept (default: 1).
        min_length (int): minimal itemset length (default: 2).
        num_dimension (int): embedding size (default: 100).
        num_epoch (int): number of passes over the whole input (default: 5).
        num_negative (int): number of negative samples (default: 5).
        initial_learning_rate (float): initial learning rate, decreasing
            linearly (default: 0.025).

    Returns:
        labels (list of object): vocabulary.
        syn0 (float32, num_label x num_dimension): first set of embeddings.
        syn1 (float32, num_label x num_dimension): second set of embeddings.

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

    train_packed_arrays(
        indices,
        offsets,
        syn0,
        syn1,
        num_epoch,
        num_negative,
        initial_learning_rate,
    )

    return labels, syn0, syn1


def train_packed_arrays(
    items,
    offsets,
    syn0,
    syn1,
    num_epoch=5,
    num_negative=5,
    initial_learning_rate=0.025,
    batch_size=64,
):
    """Train embeddings from packed arrays.

    Args:
        items (int32, num_item): itemsets, concatenated.
        offsets (int32, num_itemset + 1): boundaries in packed items.
        syn0 (float32, num_label x num_dimension): first set of embeddings.
        syn1 (float32, num_label x num_dimension): second set of embeddings.
        num_epoch (int): number of passes over the whole input (default: 5).
        num_negative (int): number of negative samples (default: 5).
        initial_learning_rate (float): initial learning rate, decreasing
            linearly (default: 0.025).
        batch_size (int): number of itemsets per batch (default: 32).

    """

    num_label, num_dimension = syn0.shape
    num_itemset = offsets.shape[0] - 1
    tmp_syn = np.empty(num_dimension, dtype=np.float32)
    indices = np.arange(num_itemset, dtype=np.int32)

    num_batch = (num_itemset - 1) // batch_size + 1
    num_step = num_epoch * num_itemset
    step = 0

    try:
        with tqdm(total=num_step) as progress:
            for epoch in range(num_epoch):

                # Shuffle itemsets
                np.random.shuffle(indices)

                # For each batch
                for batch in range(num_batch):
                    start = batch * batch_size
                    end = (batch + 1) * batch_size
                    batch_indices = indices[start:end]

                    # Learning rate decreases linearly
                    learning_rate = (1 - step / num_step) * initial_learning_rate

                    # Delegate to optimized method
                    do_unsupervised_batch(
                        items,
                        offsets,
                        batch_indices,
                        syn0,
                        syn1,
                        tmp_syn,
                        num_negative,
                        learning_rate,
                    )

                    # Update progress
                    count = batch_indices.shape[0]
                    step += count
                    progress.update(count)

    # Allow soft interruption
    except KeyboardInterrupt:
        pass
