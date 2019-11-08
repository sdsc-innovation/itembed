# -*- coding: utf-8 -*-


import collections
import numpy

from .train import train_packed_array


# Train embeddings for given itemsets
def train(itemsets, min_count=5, **kwargs):
    '''Train embeddings from collections.
    
    This helper function indexes the items, and prune less common values.
    
    For more information on the parameters, see `train_packed_array`.
    
    Args:
        itemsets (list of list of object): List of sets of hashable objects.
        min_count (int): Minimal frequency count to be kept.
        kwargs: See `train_packed_array`.
    
    Returns:
        labels (list of object): Vocabulary.
        syn0, syn1: See `train_packed_array`.
    
    Example:
        Toy dataset, where items are strings:
        
        >>> itemsets = [
        ...     ['apple', 'sugar', 'flour'],
        ...     ['pear', 'sugar', 'flour', 'butter'],
        ...     ['pear', 'apple', 'lemon']
        ... ]
        >>> labels, syn0, syn1 = train(itemsets, min_count=0, size=16)
    
    '''
    
    # Check values
    assert min_count >= 0
    
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
    
    # Train
    syn0, syn1 = train_packed_array(indices, lengths, len(labels), **kwargs)
    
    # Return relevant objects
    return labels, syn0, syn1
