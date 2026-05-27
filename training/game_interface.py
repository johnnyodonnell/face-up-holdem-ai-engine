"""The game-agnostic contract the AlphaZero pipeline depends on.

The pipeline talks only to this protocol — never the concrete game module. A
new game is added by exposing module-level functions matching `Game` plus the
two shape constants `INPUT_SIZE` and `ACTION_SIZE`. `games/holdem.py` is the
first (and currently only) implementer.

This is an N-player adaptation of the 2-player `Game` in
`tic-tac-toe-ai-engine/training/game_interface.py`. The key differences:

  * `current_seat(state)` returns a seat *index* in [0, NUM_SEATS), not a +1/-1
    sign — there are 8 actors, not 2.
  * `terminal_payoff(state, seat, start_chips_for_seat)` returns the actor's
    chip delta this hand (normalized to roughly [-1, 1]), not a tri-valued
    win/draw/loss. Each seat is its own scalar utility — there is no
    zero-sum 2-player constraint to exploit.
  * `encode(state, mover_seat)` is canonicalized to the mover's perspective:
    the network sees "me first, then opponents in seat-order starting from
    my left". One shared network plays every seat.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class Game(Protocol):
    INPUT_SIZE: int
    ACTION_SIZE: int

    def current_seat(self, state: dict) -> int:
        """Seat index (0..NUM_SEATS-1) of the actor whose turn it is, or -1
        if the state is terminal."""

    def is_terminal(self, state: dict) -> bool:
        """True iff the hand has ended (`hand_complete` or `game_over`)."""

    def legal_action_mask(self, state: dict) -> list[bool]:
        """Length-ACTION_SIZE bool list. True iff the indexed action is legal
        for the actor at `current_seat(state)`. Undefined if `is_terminal`."""

    def apply_action_idx(self, state: dict, idx: int) -> dict:
        """Return the state after the actor at `current_seat(state)` plays
        the action at index `idx`. Caller must ensure `idx` is legal."""

    def terminal_payoff(
        self, state: dict, seat: int, start_chips_for_seat: int
    ) -> float:
        """Hand-end utility for `seat`, in roughly [-1, 1]: the seat's chip
        delta from the start of the hand divided by `STARTING_STACK`, clipped.
        Defined only when `is_terminal(state)` is True."""

    def encode(self, state: dict, mover_seat: int) -> list[float]:
        """Length-INPUT_SIZE float vector, canonicalized to `mover_seat`'s
        perspective. The same `state` encoded under different `mover_seat`
        values produces different vectors — that's the point."""
