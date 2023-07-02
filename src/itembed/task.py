import numpy as np

from .optimization import do_unsupervised_batch, do_supervised_batch
from .util import index_batch_stream


class Task:
    """Abstract training task."""

    def __init__(self, learning_rate_scale):
        self.learning_rate_scale = learning_rate_scale

    def do_batch(self, learning_rate):
        """Apply training step."""

        raise NotImplementedError()


class UnsupervisedTask(Task):
    """Unsupervised training task.

    See Also
    --------
    :meth:`do_unsupervised_steps`

    Parameters
    ----------
    items: int32, num_item
        Itemsets, concatenated.
    offsets: int32, num_itemset + 1
        Boundaries in packed items.
    indices: int32, num_step
        Subset of offsets to consider.
    syn0: float32, num_label x num_dimension
        First set of embeddings.
    syn1: float32, num_label x num_dimension
        Second set of embeddings.
    weights: float32, num_item, optional
        Item weights, concatenated.
    num_negative: int32, optional
        Number of negative samples.
    learning_rate_scale: float32, optional
        Learning rate multiplier.
    batch_size: int32, optional
        Batch size.

    """

    def __init__(
        self,
        items,
        offsets,
        syn0,
        syn1,
        *,
        weights=None,
        num_negative=5,
        learning_rate_scale=1.0,
        batch_size=64,
    ):
        super().__init__(learning_rate_scale)

        # Make sure that there are no index overflow
        assert syn0.shape == syn1.shape, "synsets shape mismatch"
        assert items.min() >= 0, "negative item index"
        assert items.max() < syn0.shape[0], "out-of-bound item index"
        assert offsets.shape[0] > 1, "no itemset"
        assert offsets.min() >= 0, "negative offset"
        assert offsets.max() <= items.shape[0], "out-of-bound offset"
        assert (offsets[1:] - offsets[:-1] >= 2).all(), "itemset size must be >= 2"

        # Allocate unit weights, if needed
        if weights is None:
            weights = np.ones(items.shape[0], dtype=np.float32)
        else:
            assert weights.shape == items.shape, "weights shape mismatch"

        # Store parameters
        self.items = items
        self.weights = weights
        self.offsets = offsets
        self.syn0 = syn0
        self.syn1 = syn1
        self.num_negative = num_negative
        self.batch_size = batch_size

        # Allocate internal buffer
        size = syn0.shape[1]
        self._tmp_syn = np.empty(size, dtype=np.float32)

        # Instanciate index generator
        num_itemset = offsets.shape[0] - 1
        self.batch_iterator = index_batch_stream(num_itemset, batch_size)

    def __len__(self):
        (num_itemset,) = self.offsets.shape
        return (num_itemset - 1) // self.batch_size + 1

    def do_batch(self, learning_rate):
        indices = next(self.batch_iterator)
        do_unsupervised_batch(
            self.items,
            self.weights,
            self.offsets,
            indices,
            self.syn0,
            self.syn1,
            self._tmp_syn,
            self.num_negative,
            learning_rate * self.learning_rate_scale,
        )


class SupervisedTask(Task):
    """Supervised training task.

    See Also
    --------
    :meth:`do_supervised_steps`

    Parameters
    ----------
    left_items: int32, num_left_item
        Itemsets, concatenated.
    left_offsets: int32, num_itemset + 1
        Boundaries in packed items.
    right_items: int32, num_right_item
        Itemsets, concatenated.
    right_offsets: int32, num_itemset + 1
        Boundaries in packed items.
    left_syn: float32, num_left_label x num_dimension
        Feature embeddings.
    right_syn: float32, num_right_label x num_dimension
        Label embeddings.
    left_weights: float32, num_left_item, optional
        Item weights, concatenated.
    right_weights: float32, num_right_item, optional
        Item weights, concatenated.
    num_negative: int32, optional
        Number of negative samples.
    learning_rate_scale: float32, optional
        Learning rate multiplier.
    batch_size: int32, optional
        Batch size.

    """

    def __init__(
        self,
        left_items,
        left_offsets,
        right_items,
        right_offsets,
        left_syn,
        right_syn,
        *,
        left_weights=None,
        right_weights=None,
        num_negative=5,
        learning_rate_scale=1.0,
        batch_size=64,
    ):
        super().__init__(learning_rate_scale)

        # Make sure that there are no index overflow
        assert left_syn.shape[1] == right_syn.shape[1], "embedding size mismatch"
        assert left_items.min() >= 0, "negative item index"
        assert right_items.min() >= 0, "negative item index"
        assert left_items.max() < left_syn.shape[0], "out-of-bound item index"
        assert right_items.max() < right_syn.shape[0], "out-of-bound item index"
        assert left_offsets.shape == right_offsets.shape, "offsets shape mismatch"
        assert left_offsets.shape[0] > 1, "no itemset"
        assert right_offsets.shape[0] > 1, "no itemset"
        assert left_offsets.min() >= 0, "negative offset"
        assert right_offsets.min() >= 0, "negative offset"
        assert left_offsets.max() <= left_items.shape[0], "out-of-bound offset"
        assert right_offsets.max() <= right_items.shape[0], "out-of-bound offset"
        assert (
            left_offsets[1:] - left_offsets[:-1] >= 1
        ).all(), "itemset size must be >= 1"
        assert (
            right_offsets[1:] - right_offsets[:-1] >= 1
        ).all(), "itemset size must be >= 1"

        # Allocate unit weights, if needed
        if left_weights is None:
            left_weights = np.ones(left_items.shape[0], dtype=np.float32)
        else:
            assert left_weights.shape == left_items.shape, "weights shape mismatch"
        if right_weights is None:
            right_weights = np.ones(right_items.shape[0], dtype=np.float32)
        else:
            assert right_weights.shape == right_items.shape, "weights shape mismatch"

        # Store parameters
        self.left_items = left_items
        self.left_offsets = left_offsets
        self.left_weights = left_weights
        self.right_items = right_items
        self.right_offsets = right_offsets
        self.right_weights = right_weights
        self.left_syn = left_syn
        self.right_syn = right_syn
        self.num_negative = num_negative
        self.batch_size = batch_size

        # Allocate internal buffer
        size = left_syn.shape[1]
        self._tmp_syn = np.empty(size, dtype=np.float32)

        # Instanciate index generator
        num_itemset = left_offsets.shape[0] - 1
        self.batch_iterator = index_batch_stream(num_itemset, batch_size)

    def __len__(self):
        (num_itemset,) = self.left_offsets.shape
        return (num_itemset - 1) // self.batch_size + 1

    def do_batch(self, learning_rate):
        indices = next(self.batch_iterator)
        do_supervised_batch(
            self.left_items,
            self.left_weights,
            self.left_offsets,
            indices,
            self.right_items,
            self.right_weights,
            self.right_offsets,
            indices,
            self.left_syn,
            self.right_syn,
            self._tmp_syn,
            self.num_negative,
            learning_rate * self.learning_rate_scale,
        )


class CompoundTask(Task):
    """Group multiple sub-tasks together.

    Parameters
    ----------
    *tasks: list of Task
        Collection of tasks to train jointly.
    learning_rate_scale: float32, optional
        Learning rate multiplier.

    """

    def __init__(self, *tasks, learning_rate_scale=1.0):
        super().__init__(learning_rate_scale)
        assert len(tasks) > 0
        self.tasks = tasks

    def __len__(self):
        return max(len(task) for task in self.tasks)

    def do_batch(self, learning_rate):
        learning_rate = learning_rate * self.learning_rate_scale
        for task in self.tasks:
            task.do_batch(learning_rate)
