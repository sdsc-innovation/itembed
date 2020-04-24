
import numpy as np

from .optimization import do_unsupervised_batch, do_supervised_batch
from .util import index_batch_stream


class Task:
    """Abstract training task."""

    def __init__(self, learning_rate_scale):
        self.learning_rate_scale = learning_rate_scale

    def do_batch(self, learning_rate):
        raise NotImplementedError()


class UnsupervisedTask(Task):
    """Unsupervised training task.

    See `do_unsupervised_steps` for more information.

    """

    def __init__(self,
        items,
        offsets,
        syn0,
        syn1,
        *,
        num_negative=5,
        learning_rate_scale=1.0,
        batch_size=64,
    ):
        super().__init__(learning_rate_scale)

        # Store parameters
        self.items = items
        self.offsets = offsets
        self.syn0 = syn0
        self.syn1 = syn1
        self.num_negative = num_negative
        self.batch_size = batch_size

        # Allocate internal buffer
        size = syn0.shape[1]
        assert syn1.shape[1] == size
        self._tmp_syn = np.empty(size, dtype=np.float32)

        # Instanciate index generator
        num_itemset, = offsets.shape
        self.batch_iterator = index_batch_stream(num_itemset, batch_size)

    def __len__(self):
        num_itemset, = self.offsets.shape
        return (num_itemset - 1) // self.batch_size + 1

    def do_batch(self, learning_rate):
        indices = next(self.batch_iterator)
        do_unsupervised_batch(
            self.items,
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

    See `do_supervised_steps` for more information.

    """

    def __init__(self,
        left_items,
        left_offsets,
        right_items,
        right_offsets,
        left_syn,
        right_syn,
        *,
        num_negative=5,
        learning_rate_scale=1.0,
        batch_size=64,
    ):
        super().__init__(learning_rate_scale)

        # Store parameters
        self.left_items = left_items
        self.left_offsets = left_offsets
        self.right_items = right_items
        self.right_offsets = right_offsets
        self.left_syn = left_syn
        self.right_syn = right_syn
        self.num_negative = num_negative
        self.batch_size = batch_size

        # Allocate internal buffer
        size = left_syn.shape[1]
        assert right_syn.shape[1] == size
        self._tmp_syn = np.empty(size, dtype=np.float32)

        # Instanciate index generator
        num_itemset, = left_offsets.shape
        assert right_offsets.shape[0] == num_itemset
        self.batch_iterator = index_batch_stream(num_itemset, batch_size)

    def __len__(self):
        num_itemset, = self.left_offsets.shape
        return (num_itemset - 1) // self.batch_size + 1

    def do_batch(self, learning_rate):
        indices = next(self.batch_iterator)
        do_supervised_batch(
            self.left_items,
            self.left_offsets,
            indices,
            self.right_items,
            self.right_offsets,
            indices,
            self.left_syn,
            self.right_syn,
            self._tmp_syn,
            self.num_negative,
            learning_rate * self.learning_rate_scale,
        )


class CompoundTask(Task):
    """Group multiple sub-tasks together."""

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
