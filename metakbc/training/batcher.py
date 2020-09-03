# -*- coding: utf-8 -*-

import numpy as np

from metakbc.util import make_batches


class Batcher:
    def __init__(self,
                 nb_examples: int,
                 batch_size: int,
                 nb_epochs: int,
                 random_state: np.random.RandomState) -> None:
        self.nb_examples = nb_examples
        self.batch_size = batch_size
        self.random_state = random_state

        size = nb_epochs * self.nb_examples
        self.curriculum = np.zeros(size, dtype=np.int32)

        for epoch_no in range(nb_epochs):
            start = epoch_no * self.nb_examples
            end = (epoch_no + 1) * self.nb_examples
            self.curriculum[start:end] = self.random_state.permutation(self.nb_examples)

        self.batches = make_batches(self.curriculum.shape[0], batch_size)
        self.nb_batches = len(self.batches)

    def get_batch(self, batch_start: int, batch_end: int) -> np.ndarray:
        return self.curriculum[batch_start:batch_end]
