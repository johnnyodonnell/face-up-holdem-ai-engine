"""FIFO replay buffer for self-play training records.

Each record is a triple:
    (encoded_input[INPUT_SIZE], policy_target[ACTION_SIZE], value_target[NUM_SEATS])

The policy_target is the MCTS visit-count distribution at this decision
(already softened by temperature where appropriate). The value_target is
the per-seat payoff vector in the recording actor's mover frame: slot 0
holds the value for the recording actor, slot 1 for the seat to its left,
etc. All 8 slots are supervised — we know every seat's chip delta at the
end of a hand, so why throw away 7/8 of the signal.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, encoded: list[float], policy: np.ndarray, value: np.ndarray) -> None:
        self._buf.append((
            np.asarray(encoded, dtype=np.float32),
            policy.astype(np.float32),
            value.astype(np.float32),
        ))

    def add_many(
        self,
        records: Iterable[tuple[list[float], np.ndarray, np.ndarray]],
    ) -> None:
        for r in records:
            self.add(*r)

    def sample(
        self, batch_size: int, rng: random.Random
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = min(batch_size, len(self._buf))
        idxs = [rng.randrange(len(self._buf)) for _ in range(k)]
        inputs = np.stack([self._buf[i][0] for i in idxs])
        policies = np.stack([self._buf[i][1] for i in idxs])
        values = np.stack([self._buf[i][2] for i in idxs])
        return inputs, policies, values
