# -*- coding: utf-8 -*-


import math
from numba import jit, float32
import numpy
import random
from tqdm import tqdm


# TODO maybe add RNG as parameter?


# Train embedding using concatenated arrays
def train_packed_array(indices, lengths, num_labels=None, size=100, num_epochs=5, num_negatives=5, starting_alpha=0.025):
    '''Train embeddings from packed indexed itemsets.
    
    For compactness and performance reasons, itemsets are stored in a one-
    dimensional array. An additional array is therefore used to identify
    boundaries, by specifying the length of each itemset.
    
    This uses skip-gram with negative sampling, as defined by Mikolov et al.
    in word2vec. Default parameters are taken from official implementation.
    
    Args:
        indices (int32, N): Item indices of the M itemsets, concatenated.
        lengths (int32, M): Length of each itemset, concatenated.
        num_labels (int32): Number of items (default `max(indices)+1`).
        size (int32): Embedding size (default 100).
        num_epochs (int32): Number of passes over the whole input (default 5).
        num_negatives (int32): Number of negative sample per item (default 5).
        starting_alpha (float32): Initial learning rate, decaying linearly to
            zero (default 0.025).
    
    Returns:
        syn0 (float32, num_labels x size): First set of embeddings.
        syn1 (float32, num_labels x size): Second set of embeddings.
    
    Example:
        Toy dataset, specified directly as packed array.
        
        >>> indices = np.array([
        ...     0, 3, 7, 2,
        ...     1, 2,
        ...     4, 5, 6, 0, 1
        ... ])
        >>> lengths = np.array([
        ...     4,
        ...     2,
        ...     5
        ... ])
        >>> syn0, syn1 = train_packed_array(
        ...     indices, lengths,
        ...     num_labels = 8,
        ...     size = 32,
        ... )
    
    '''
    
    # TODO accept weight at some point?
    # TODO accept hierarchy somehow (i.e. ancestors/descendants are updated as well)
    
    # Fill missing parameters
    if num_labels is None:
        if indices.size == 0:
            num_labels = 1
        else:
            num_labels = indices.max() + 1
    
    # Check values
    assert size > 0
    assert num_epochs > 0
    assert num_negatives >= 0
    assert starting_alpha > 0
    
    # Initial first set of embeddings
    syn0 = numpy.random.rand(num_labels, size).astype(numpy.float32)
    syn0 -= 0.5
    syn0 /= size
    
    # Complementary embedding set is initialized to 0
    syn1 = numpy.zeros((num_labels, size), dtype=numpy.float32)
    
    # Internal buffer is also allocated
    tmp_syn0 = numpy.empty(size, dtype=numpy.float32)
    
    # Do training
    _train(indices, lengths, syn0, syn1, tmp_syn0, num_epochs, num_negatives, starting_alpha)
    
    # Return relevant objects
    return syn0, syn1


# Train using preprocess itemsets
def _train(indices, lengths, syn0, syn1, tmp_syn0, num_epochs, num_negatives, starting_alpha):
    num_labels, size = syn0.shape
    
    # For each epoch
    # TODO add parameter to control/disable tqdm
    with tqdm(total=num_epochs * indices.shape[0]) as progress:
        for epoch in range(num_epochs):
            
            # For each context
            offset = 0
            for i in range(lengths.shape[0]):
                length = lengths[i]
                
                # Update learning rate
                alpha = (1 - offset / (indices.shape[0] + 1)) * starting_alpha
                
                # Apply optimized step
                do_step(indices, offset, length, syn0, syn1, tmp_syn0, num_negatives, alpha)
                
                # Move to next context
                offset += length
                progress.update(length)


# Optimized training step
@jit(nopython=True)
def do_step(indices, offset, length, syn0, syn1, tmp_syn0, num_negatives, alpha):
    '''Apply a single training step.
    
    Args:
        indices (int32, N): Item index array.
        offset (int32): Start of itemset (inclusive) in array.
        length (int32): Number of items in itemset.
        syn0 (float32, num_labels x size): First set of embeddings.
        syn1 (float32, num_labels x size): Second set of embeddings.
        tmp_syn0 (float32, size): Internal buffer (allocated only once, for
            performance).
        num_negatives (int32): Number of negative samples.
        alpha (float32): Learning rate.
    
    '''
    num_labels, size = syn0.shape
    
    # Enumerate words
    for j in range(offset, offset + length):
        word = indices[j]
        
        # For each neighbor
        for k in range(offset, offset + length):
            if j != k:
                neighbor_word = indices[k]
                
                # Approximate softmax update
                tmp_syn0[:] = 0
                for n in range(num_negatives + 1):
                    
                    # Apply a single positive update
                    if n == 0:
                        target = word
                        label = 1
                    
                    # And several negative updates
                    else:
                        target = random.randint(0, num_labels - 1) # TODO check that it is not equal to true label
                        label = 0
                    
                    # Compute dot product between reference and target
                    f = numpy.dot(syn0[neighbor_word], syn1[target])
                    
                    # Compute gradient
                    g = (label - expit(f)) * alpha
                    
                    # Backpropagate
                    for c in range(size):
                        tmp_syn0[c] += g * syn1[target, c]
                    for c in range(size):
                        syn1[target, c] += g * syn0[neighbor_word, c]
                for c in range(size):
                    syn0[neighbor_word, c] += tmp_syn0[c]


# Logistic function
@jit(float32(float32), nopython=True)
def expit(x):
    '''Compute logistic activation.
    
    '''
    return 1 / (1 + math.exp(-x))
