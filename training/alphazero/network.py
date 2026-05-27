"""The policy/value network — a small MLP, mover-frame canonicalized.

Input  : INPUT_SIZE-dim float vector from games.holdem.encode().
Output : (policy_logits[ACTION_SIZE], value_vec[NUM_SEATS] each in [-1, 1] via tanh).

The value head is an **N-vector**, not a scalar. Hold'em is 8-player; in a
non-2-player game the "opponent's value = -mine" identity that 2-player
zero-sum MCTS implementations exploit doesn't hold. We instead predict a
value for every seat, in mover-frame slot order: `value[0]` is the value
for the encoded mover, `value[1]` for the seat at the mover's left, etc.

forward() returns *raw* policy logits — softmax and illegal-move masking
are done by the caller (MCTS at inference, the loss function at training
time). This keeps the network identical to the hand-written JS forward
pass that ships in the browser, which the parity check relies on.

Architecturally based on fox-lite-ai-engine/training/alphazero/network.py;
the value-head shape is the main adaptation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from games.holdem import INPUT_SIZE, ACTION_SIZE, NUM_SEATS


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        action_size: int = ACTION_SIZE,
        num_seats: int = NUM_SEATS,
    ):
        super().__init__()
        hidden = config.HIDDEN_SIZE

        trunk: list[nn.Module] = []
        prev = input_size
        for _ in range(config.TRUNK_LAYERS):
            trunk.append(nn.Linear(prev, hidden))
            prev = hidden
        self.trunk = nn.ModuleList(trunk)

        self.policy_head = nn.Linear(hidden, action_size)
        self.value_head = nn.Linear(hidden, num_seats)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [batch, input_size]; returns (policy_logits[..., 7], value[..., 8]).

        Both bounded outputs are produced raw; the value is then tanh'd so
        each per-seat scalar lives in (-1, 1) — matches the chip-delta /
        STARTING_STACK target produced by `games.holdem.terminal_payoff`.
        """
        for layer in self.trunk:
            x = F.relu(layer(x))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy_logits, value


@torch.no_grad()
def infer(net: PolicyValueNet, encoded: list[float]) -> tuple[list[float], list[float]]:
    """Single-state inference returning raw logits + value vector as Python types.

    The input adopts the network's dtype, so a net moved to float64 (via
    `.double()`, used in parity checks) runs the whole forward pass in
    float64 to match the JS engine.
    """
    dtype = next(net.parameters()).dtype
    device = next(net.parameters()).device
    x = torch.tensor([encoded], dtype=dtype, device=device)
    logits, value = net(x)
    return logits[0].tolist(), value[0].tolist()


@torch.no_grad()
def infer_batch(
    net: PolicyValueNet, encoded_batch: list[list[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    """Batched inference. Returns (policies, values) where each is a list of
    Python float-lists. policies[i] is length ACTION_SIZE, values[i] is
    length NUM_SEATS."""
    if not encoded_batch:
        return [], []
    dtype = next(net.parameters()).dtype
    device = next(net.parameters()).device
    x = torch.tensor(encoded_batch, dtype=dtype, device=device)
    logits, value = net(x)
    return logits.tolist(), value.tolist()
