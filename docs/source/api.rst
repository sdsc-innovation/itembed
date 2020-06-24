.. _api:


Developer Interface
===================

.. module:: itembed

This part of the documentation covers the public interface of itembed.


Preprocessing Tools
-------------------

A few helpers are provided to clean the data and convert to the expected
format.

.. autofunction:: index_batch_stream
.. autofunction:: pack_itemsets
.. autofunction:: prune_itemsets


Tasks
-----

Tasks are high-level building blocks used to define an optimization problem.

.. autoclass:: Task
   :members:
   :undoc-members:

.. autoclass:: UnsupervisedTask
   :inherited-members:
   :members:
   :undoc-members:

.. autoclass:: SupervisedTask
   :inherited-members:
   :members:
   :undoc-members:

.. autoclass:: CompoundTask
   :inherited-members:
   :members:
   :undoc-members:


Training Tools
--------------

Embeddings initialization and training loop helpers:

.. autofunction:: initialize_syn
.. autofunction:: train


Postprocessing Tools
--------------------

Once embeddings are trained, some methods are provided to normalize and use them.

.. autofunction:: softmax
.. autofunction:: norm
.. autofunction:: normalize


Low-Level Optimization Methods
------------------------------

At its core, itembed is a set of optimized methods.

.. autofunction:: expit
.. autofunction:: do_step
.. autofunction:: do_supervised_steps
.. autofunction:: do_unsupervised_steps
.. autofunction:: do_supervised_batch
.. autofunction:: do_unsupervised_batch
