# User interface

This part of the documentation covers the public interface of itembed.


## Preprocessing tools

A few helpers are provided to clean the data and convert to the expected
format.

### ::: itembed.index_batch_stream
### ::: itembed.pack_itemsets
### ::: itembed.prune_itemsets


## Tasks

Tasks are high-level building blocks used to define an optimization problem.

### ::: itembed.Task
### ::: itembed.UnsupervisedTask
### ::: itembed.SupervisedTask
### ::: itembed.CompoundTask


## Training tools

Embeddings initialization and training loop helpers:

### ::: itembed.initialize_syn
### ::: itembed.train


## Postprocessing tools

Once embeddings are trained, some methods are provided to normalize and use them.

### ::: itembed.softmax
### ::: itembed.norm
### ::: itembed.normalize


## Low-level optimization methods

At its core, itembed is a set of optimized methods.

### ::: itembed.expit
### ::: itembed.do_step
### ::: itembed.do_supervised_steps
### ::: itembed.do_unsupervised_steps
### ::: itembed.do_supervised_batch
### ::: itembed.do_unsupervised_batch
