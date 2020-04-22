
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
        num_negative (int32): number of negative samples.
        learning_rate (int32): learning rate.

    """

    num_right, num_dimension = syn_right.shape

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

    For each item, a single random neighbor is sampled to define a pair. This
    means that only a subset of possible pairs is considered. The reason is
    twofold: training stays in linear complexity w.r.t. itemset lengths and
    large itemsets do not dominate smaller ones.

    Itemset must have at least 2 items. Length is not checked, for efficiency.

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
            num_negative, learning_rate
        )


@jit(
    void(
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
def do_supervised_steps(
    left_itemset,
    right_itemset,
    left_syn,
    right_syn,
    tmp_syn,
    num_negative,
    learning_rate,
):
    """Apply steps from two itemsets.

    This is used in a supervised setting, where left-hand items are features
    and right-hand items are labels.

    Args:
        left_itemset (int32, left_length): feature items.
        right_itemset (int32, right_length): label items.
        left_syn (float32, num_left_label x num_dimension): feature embeddings.
        right_syn (float32, num_right_label x num_dimension): label embeddings.
        tmp_syn (float32, num_dimension): internal buffer (allocated only once,
            for performance).
        num_negative (int32): number of negative samples.
        learning_rate (float32): learning rate.

    """

    # For each pair
    # TODO maybe need to apply subsampling?
    # TODO possibly two passes, to garantee that each item set is fully used?
    for left in left_itemset:
        for right in right_itemset:

            # Apply update
            do_step(
                left, right,
                left_syn, right_syn, tmp_syn,
                num_negative, learning_rate
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
    """Apply unsupervised steps from multiple itemsets.

    See `do_unsupervised_steps` for more information.

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

    for i in indices:
        do_unsupervised_steps(
            items[offsets[i]:offsets[i+1]],
            syn0,
            syn1,
            tmp_syn,
            num_negative,
            learning_rate,
        )


@jit(
    void(
        int32[:],
        int32[:],
        int32[:],
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
def do_supervised_batch(
    left_items,
    left_offsets,
    left_indices,
    right_items,
    right_offsets,
    right_indices,
    left_syn,
    right_syn,
    tmp_syn,
    num_negative,
    learning_rate,
):
    """Apply supervised steps from multiple itemsets.

    See `do_supervised_steps` for more information.

    Args:
        left_items (int32, num_item): itemsets, concatenated.
        left_offsets (int32, num_itemset + 1): boundaries in packed items.
        left_indices (int32, num_step): subset of offsets to consider.
        right_items (int32, num_item): itemsets, concatenated.
        right_offsets (int32, num_itemset + 1): boundaries in packed items.
        right_indices (int32, num_step): subset of offsets to consider.
        left_syn (float32, num_left_label x num_dimension): feature embeddings.
        right_syn (float32, num_right_label x num_dimension): label embeddings.
        tmp_syn (float32, num_dimension): internal buffer (allocated only once,
            for performance).
        num_negative (int32): number of negative samples.
        learning_rate (float32): learning rate.

    """

    length = left_indices.shape[0]
    for i in range(length):
        j = left_indices[i]
        k = right_indices[i]
        do_supervised_steps(
            left_items[left_offsets[j]:left_offsets[j+1]],
            right_items[right_offsets[k]:right_offsets[k+1]],
            left_syn,
            right_syn,
            tmp_syn,
            num_negative,
            learning_rate,
        )
