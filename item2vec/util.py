
from collections import Counter

import numpy as np

from tqdm import tqdm


def index_batch_stream(num_index, batch_size):
    """Indices generator."""

    assert num_index > 0
    assert batch_size > 0

    indices = np.arange(num_index, dtype=np.int32)
    # TODO improve this algorithm

    # Replicate indices, to cover at least one batch
    if num_index < batch_size:
        repeat = batch_size // num_index + 1
        indices = np.tile(indices, repeat)
        num_index *= repeat

    # Loop forever
    while True:
        np.random.shuffle(indices)
        i = 0
        while i + batch_size <= num_index:
            yield indices[i:i+batch_size]
            i += batch_size


def pack_itemsets(itemsets, *, min_count=1, min_length=2):
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


def prune_itemsets(indices, offsets, *, mask=None, min_length=None):
    """Filter packed indices.

    Either an explicit mask or a length threshold must be defined, but both
    cannot be provided at the same time.

    Args:
        indices (int32, num_item): packed index array.
        offsets (int32, num_itemset + 1): itemsets offsets in packed array.
        mask (bool, num_itemset): boolean mask.
        min_length (int): minimum length, inclusive.

    Returns:
        indices (int32, num_item): packed index array.
        offsets (int32, num_itemset + 1): itemsets offsets in packed array.

    """

    # Build mask from length limit, if needed
    lengths = offsets[1:] - offsets[:-1]
    if min_length is not None:
        assert mask is None
        mask = lengths >= min_length
    assert lengths.shape == mask.shape

    # Allocate buffers
    filtered_indices = np.zeros(lengths[mask].sum(), dtype=np.int32)
    filtered_offsets = np.zeros(mask.sum() + 1, dtype=np.int32)

    # Build new itemsets
    offset = 0
    j = 0
    for i in range(len(mask)):
        keep = mask[i]
        if keep:
            length = lengths[i]
            filtered_indices[offset:offset+length] = indices[offsets[i]:offsets[i+1]]
            offset += length
            filtered_offsets[j] = offset
            j += 1
    return filtered_indices, filtered_offsets


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
    task,
    *,
    num_epoch=10,
    initial_learning_rate=0.025,
):
    """Train loop.

    Learning rate decreases linearly, down to zero.

    Keyboard interruption are silently captured, which interrupt the training
    process.

    A progress bar is shown, using `tqdm`.

    Args:
        task (Task): top-level task to train.
        num_epoch (int): number of passes across the whole task (default: 10).
        initial_learning_rate (float): maximum learning rate (default: 0.025).

    """

    try:

        # Iterate over epochs and batches
        num_batch = len(task)
        num_step = num_epoch * num_batch
        step = 0
        with tqdm(total=num_step) as progress:
            for epoch in range(num_epoch):
                for batch in range(num_batch):

                    # Learning rate decreases linearly
                    learning_rate = (1 - step / num_step) * initial_learning_rate

                    # Apply step
                    task.do_batch(learning_rate)

                    # Update progress
                    step += 1
                    progress.update(1)

    # Allow soft interruption
    except KeyboardInterrupt:
        pass


def softmax(x):
    """Compute softmax."""

    e = np.exp(x)
    return e / e.sum(axis=-1)[..., None]
