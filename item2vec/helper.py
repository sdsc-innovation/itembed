
import numpy as np

from tqdm import tqdm


def train(
    task,
    *,
    num_epoch=5,
    initial_learning_rate=0.025,
):

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
