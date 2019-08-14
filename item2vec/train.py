# -*- coding: utf-8 -*-


import collections
import math
from numba import jit, float32
import numpy
import random
from tqdm import tqdm


# Train embeddings for given itemsets
def train(itemsets, size=128, min_count=5, num_epochs=5, num_negatives=5, starting_alpha=0.025):
    
    # Check values
    assert size > 0
    assert min_count >= 0
    assert num_epochs > 0
    assert num_negatives >= 0
    assert starting_alpha > 0
    
    # Count labels
    counter = collections.Counter()
    for itemset in itemsets:
        counter.update(itemset)
    
    # Define label list
    labels = [label for label, count in counter.most_common() if count >= min_count]
    label_map = {label : i for i, label in enumerate(labels)}
    
    # Generate indices
    all_indices = []
    for itemset in itemsets:
        # TODO accept weight
        indices = []
        for label in itemset:
            label = label_map.get(label)
            if label is not None:
                indices.append(label)
        if len(indices) > 1:
            all_indices.append(indices)
    
    # Pack as flat array
    lengths = numpy.array([len(indices) for indices in all_indices], dtype=numpy.intp)
    indices = numpy.array([index for indices in all_indices for index in indices], dtype=numpy.intp)
    assert len(indices) > 0
    
    # Initialize parameters
    syn0 = (numpy.random.rand(len(labels), size).astype(dtype=numpy.float32) - 0.5) / size
    syn1 = numpy.zeros((len(labels), size), dtype=numpy.float32)
    tmp_syn0 = numpy.zeros(size, dtype=numpy.float32)
    
    # Train using optimized method
    _train(indices, lengths, syn0, syn1, tmp_syn0, len(labels), num_epochs, num_negatives, starting_alpha)
    
    # Return relevant objects
    return labels, syn0, syn1


# Train using preprocess itemsets
def _train(indices, lengths, syn0, syn1, tmp_syn0, num_labels, num_epochs, num_negatives, starting_alpha):
    size = syn0.shape[1]
    
    # For each epoch
    with tqdm(total=num_epochs * indices.shape[0]) as progress:
        for epoch in range(num_epochs):
            
            # For each context
            offset = 0
            for i in range(lengths.shape[0]):
                length = lengths[i]
                
                # Update learning rate
                alpha = (1 - offset / (indices.shape[0] + 1)) * starting_alpha
                
                # Apply optimized step
                _step(indices, lengths, syn0, syn1, tmp_syn0, offset, length, alpha, size, num_labels, num_negatives)
                
                # Move to next context
                offset += length
                progress.update(length)


@jit(nopython=True)
def _step(indices, lengths, syn0, syn1, tmp_syn0, offset, length, alpha, size, num_labels, num_negatives):
    
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


@jit(float32(float32), nopython=True)
def expit(x):
    return 1 / (1 + math.exp(-x))
