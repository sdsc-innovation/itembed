
import math
import random

import numpy as np

from numba import jit, float32, int32, void


@jit(
    float32(float32),
    nopython=True,
    nogil=True,
    fastmath=True,
)
def expit(x):
    """Compute logistic activation."""

    return 1 / (1 + math.exp(-x))


@jit(
    void(
        int32,
        int32,
        float32[:, ::1],
        float32[:, ::1],
        float32[::1],
        int32,
        int32,
        int32,
        float32,
    ),
    nopython=True,
    nogil=True,
    fastmath=True,
)
def do_step(
    left,
    right,
    syn_left,
    syn_right,
    tmp_syn,
    num_right,
    num_dimension,
    num_negative,
    learning_rate,
):
    """Apply a single training step.

    Args:
        left (int32): left-hand item.
        right (int32): right-hand item.
        syn_left (float32, num_left x num_dimension): left-hand embeddings.
        syn_right (float32, num_right x num_dimension): right-hand embeddings.
        tmp_syn (float32, num_dimension): internal buffer (allocated only once,
            for performance).
        num_right (int32): number of right-hand labels.
        num_dimension (int32): embedding size.
        num_negative (int32): number of negative samples.
        learning_rate (int32): learning rate.

    """

    # Approximate softmax update
    tmp_syn[:] = 0
    for n in range(num_negative + 1):

        # Apply a single positive update
        if n == 0:
            target = right
            label = 1

        # And several negative updates
        else:
            target = random.randint(0, num_right - 1)
            label = 0

        # Compute dot product between reference and target
        logit = np.dot(syn_left[left], syn_right[target])

        # Compute gradient scale
        gradient = (label - expit(logit)) * learning_rate

        # Accumulate gradients for left-hand embeddings
        for c in range(num_dimension):
            tmp_syn[c] += gradient * syn_right[target, c]

        # Backpropagate to right-hand embeddings
        for c in range(num_dimension):
            syn_right[target, c] += gradient * syn_left[left, c]

    # Backpropagate to left-hand embeddings
    for c in range(num_dimension):
        syn_left[left, c] += tmp_syn[c]


@jit(
    void(
        int32[:],
        float32[:, ::1],
        float32[:, ::1],
        float32[::1],
        int32,
        float32,
    ),
    nopython=True,
    nogil=True,
    fastmath=True,
)
def do_unsupervised_steps(
    itemset,
    syn0,
    syn1,
    tmp_syn,
    num_negative,
    learning_rate,
):
    """Apply steps from a single itemset.

    This is used in an unsupervised setting, where co-occurrence is used as a
    knowledge source. It follows the skip-gram method, as introduced by Mikolov
    et al.

    Args:
        itemset (int32, length): items.
        syn0 (float32, num_label x num_dimension): first set of embeddings.
        syn1 (float32, num_label x num_dimension): second set of embeddings.
        tmp_syn (float32, num_dimension): internal buffer (allocated only once,
            for performance).
        num_negative (int32): number of negative samples.
        learning_rate (float32): learning rate.

    """

    num_label, num_dimension = syn0.shape
    length = itemset.shape[0]

    # Enumerate words
    for j in range(length):
        left = itemset[j]

        # Choose a single random neighbor
        k = random.randint(0, length - 2)
        if k >= j:
            k += 1
        right = itemset[k]

        # Apply update
        do_step(
            left, right,
            syn0, syn1, tmp_syn,
            num_label, num_dimension, num_negative,
            learning_rate
        )


@jit(
    void(
        int32[:],
        int32[:],
        int32[:],
        float32[:, ::1],
        float32[:, ::1],
        float32[::1],
        int32,
        float32,
    ),
    nopython=True,
    nogil=True,
    fastmath=True,
)
def do_unsupervised_batch(
    items,
    offsets,
    indices,
    syn0,
    syn1,
    tmp_syn,
    num_negative,
    learning_rate,
):
    """Apply steps from multiple itemsets.

    This is used in an unsupervised setting, where co-occurrence is used as a
    knowledge source. It follows the skip-gram method, as introduced by Mikolov
    et al.

    Args:
        items (int32, num_item): itemsets, concatenated.
        offsets (int32, num_itemset + 1): boundaries in packed items.
        indices (int32, num_step): subset of offsets to consider.
        syn0 (float32, num_label x num_dimension): first set of embeddings.
        syn1 (float32, num_label x num_dimension): second set of embeddings.
        tmp_syn (float32, num_dimension): internal buffer (allocated only once,
            for performance).
        num_negative (int32): number of negative samples.
        learning_rate (float32): learning rate.

    """

    num_itemset = indices.shape[0]
    for index in indices:
        do_unsupervised_steps(
            items[offsets[index]:offsets[index+1]],
            syn0,
            syn1,
            tmp_syn,
            num_negative,
            learning_rate,
        )
