# Future plan — neural network engine via self-play

Replace the heuristic bot in `src/engine/heuristic.js` with a trained
policy/value network. The training pipeline already exists end-to-end on
`asus-nvidia` (Phase 3 smoke-test passed); what remains is a real training
run and browser deployment.

This document captures the path forward and the rationale for taking it,
so we can pick the work back up later without re-deriving any of it.

## Status

| Phase | What | State |
| --- | --- | --- |
| 1 | Python rules port + N-player `Game` adapter + mover-frame encoder + JS↔Python parity harness | **done** — 5038 rules transitions verified byte-identical |
| 2 | Equity-based heuristic baseline (both Python + JS) + parity check + sanity check | **done** — heuristic beats random by +982 BB/100, JS engine swapped in for the browser |
| 3 | AlphaZero core: `PolicyValueNet` with 8-vector value head, MCTS with chance nodes for the flop/turn/river deals, self-play loop, BB/100 arena gate, training driver | **done** — smoke run on GB10 completed (iter 1 losses → iter 2 losses → arena PROMOTED) |
| 4 | Real training run + heuristic-evaluator script + hyperparameter iteration | **pending** |
| 5 | Browser integration: `nn.js` forward pass, `weights.json` export, JS↔Python float64 parity check, swap `App.jsx` import | **pending** |

All code from phases 1–3 is committed on `origin/main`. The training
pipeline runs on `asus-nvidia` in `~/Code/face-up-holdem-ai-engine/training`
with a venv that has the cu130 nightly PyTorch wheel installed.

## Why a neural network will outperform the heuristic

Face-up Hold'em is unusual: because every hole card is visible, a lot of
what makes general poker AI hard (hand-reading, range estimation, bluff
construction) is reduced to a mechanical equity calculation. Our heuristic
does that calculation correctly and adds pot-odds-based folding. Bluffing
is largely off the table — if I bet big with 72o, opponents see the 72o
and call or fold based on their *actual* current equity, not on what they
think I'm representing.

That said, the heuristic has real structural limitations that a self-play
NN can plausibly close. We're confident the NN will be stronger — not
because of bluffing, but because of the items below — provided training
is done correctly.

### 1. Opponent modelling for fold equity (non-deceptive)

The heuristic computes equity assuming every live opponent stays in to
showdown. Opponents in practice fold to pressure, which changes EV in
ways the heuristic structurally can't see.

Worked example: 8-handed preflop with TT. Hero's raw equity against all
seven live opponents is ~30%, so the heuristic just calls or folds. But TT
is actually a profitable 3-bet — it folds out four or five opponents whose
hands can't continue, leaving hero heads-up or 3-way with ~50% equity plus
all the dead money from the blinds and limps. The NN learns this through
self-play; the heuristic can't, because its equity calc treats opponents
as inert.

This is **not** bluffing — hero isn't representing a hand they don't have.
It's correctly accounting for the fact that opponents will fold dominated
hands, which the heuristic ignores.

### 2. Bet sizing

The heuristic picks a raise size from one variable (equity bucket: ≥0.50 →
½pot, ≥0.65 → pot, ≥0.80 → 2pot, ≥0.90 → all-in). Optimal sizing depends
on opponent calling tendencies, stack-to-pot ratio, board texture, and
position — a function of state the NN can fit.

Even within "value-bet with a strong hand", the right size varies: against
opponents that call wide, bigger is better; against tight opponents,
smaller extracts more. The heuristic ignores all of this.

### 3. Multi-street planning

The heuristic decides each street in isolation. MCTS at training time lets
the NN reason about future-street EV — keep the pot small now to maintain
implied odds, or build the pot now to maximize value on later streets.
This planning is genuinely available in face-up Hold'em (chance still
exists in the form of unrevealed board cards).

### 4. Equity-denial pricing

When hero has 60% with a draw out against them, the heuristic raises
½pot. But maybe shoving is correct because it locks in equity that would
otherwise leak — opponent's draw becomes incorrect to call. The heuristic
doesn't optimize for "what size makes opponent's continue wrong"; it just
maps equity to size. The NN can learn this.

### 5. Position and stack-depth adjustments

The heuristic uses the same pot-fractions regardless of position or stack
depth. Late-position has an inherent edge (more info before deciding);
short stacks change pot dynamics; deep stacks reward speculative hands.
The NN can learn positional and SPR-aware strategy.

## Realistic strength estimate

Modest. Probably **+20 to +80 BB/100 with solid training**, with high
uncertainty either direction. Not the +982 BB/100 that the heuristic
achieved against random — random play is uniquely terrible. The
heuristic-vs-NN gap will be measured in tens of big blinds per hundred
hands, not hundreds.

A few framings for what this looks like:

- **In chip-EV terms:** noticeable but not transformative. Tens of BB/100
  is real money in a real game.
- **In feel:** the bots will size more thoughtfully, occasionally check-
  raise, 3-bet hands the heuristic just calls, and shove with a logic
  that's harder to predict than "equity threshold crossed".
- **In comparison to standard Hold'em:** Pluribus famously crushed pros at
  six-handed no-limit. The face-up gap is intrinsically smaller because
  most of what made Pluribus revolutionary (hand-reading + bluffing under
  hidden info) doesn't apply here. We're closing the smaller gap that
  remains.

## What "done correctly" means

The confidence above assumes the training run actually converges. The
ways it could fail to:

- **Value head divergence.** AlphaZero loops occasionally regress
  silently. The arena gate filters this, but only with enough hands per
  match for a tight CI. Plan default: `ARENA_MATCHES=8 ×
  ARENA_HANDS_PER_MATCH=200 = 1600` hands per arena, with a `+3 BB/100`
  promotion threshold. Tune if regressions slip through.
- **Insufficient self-play volume.** Hold'em has a vastly larger state
  space than tic-tac-toe; underfitting is a real risk. Plan
  `NUM_ITERATIONS=40 × GAMES_PER_ITER=64` is a starting point and will
  likely need to grow.
- **Hyperparameter drift.** Learning rate, value-loss weight, Dirichlet
  noise, MCTS sim count — all need tuning. Phase 4 budgets time for this.
- **Replay buffer staleness.** `BUFFER_SIZE=500000` may need to expand
  with `GAMES_PER_ITER`, otherwise old self-play data dominates.
- **Chance-node sample count too low.** `CHANCE_SAMPLES=4` per decision
  is a starting guess; if value estimates feel noisy at street boundaries,
  bump it.

## Remaining work

### Phase 4 — real training run

- `training/scripts/evaluate_heuristic.py` — head-to-head: best.pt seat vs
  heuristic seats, many hands, BB/100 with CI. The absolute-strength
  yardstick for the trained network. The arena gate only proves
  "monotonic improvement"; this script proves "actually any good".
- Real training run on GB10, target: NN beats heuristic by ≥+5 BB/100 with
  non-overlapping CI. Iterate on hyperparameters until that's true.
- Expected wall time: hours to a few days of training, plus hours of
  hyperparameter iteration.

### Phase 5 — browser integration

- `training/scripts/export_weights.py` — write `best.pt` to
  `src/engine/weights.json` (schema mirrors fox-lite: `meta`, `trunk`,
  `policyHead`, `valueHead`).
- `src/engine/nn.js` — generic forward pass (`linear()` + `forward()`).
  Can be lifted from fox-lite's `nn.js` almost verbatim — it's
  game-agnostic.
- `src/engine/nnGame.js` — Hold'em-specific encoder, byte-for-byte
  mirror of `games/holdem.py`'s `encode()`. The encoder is the
  highest-stakes parity point.
- `src/engine/neural.js` — implements `bestMove(state)`: encode, forward
  pass, mask illegal actions, argmax over policy logits, map action
  index back to `{type, amount?}`.
- Swap the import in `src/App.jsx` from `./engine/heuristic.js` to
  `./engine/neural.js`.
- Parity harness:
  - `training/scripts/network_parity_dump.py` + `network_parity_check.mjs`
    — float64 forward-pass agreement to ~1e-12.
  - Confirm `parity_check.py` still passes (rules port unaffected).
- Manual verification: `npm run dev`, play a few hands, confirm bots
  decide in <100ms and behave visibly differently from the heuristic
  (different sizings, occasional check-raises, etc.).

Expected wall time: 3–5 days.

## Reference points

- Plan file from the original investigation:
  `/home/johnny/.claude/plans/polished-floating-crane.md` (more detail on
  state-encoding dimensions, action-discretization rationale, and the
  fox-lite template patterns we're reusing)
- Sibling project we're mirroring:
  `/home/johnny/Code/fox-lite-ai-engine` (AlphaZero pipeline already
  trained and shipping in a browser)
- Compute: `asus-nvidia` (NVIDIA GB10, 119 GiB unified memory, CUDA 13).
  Project clone + venv already provisioned at
  `~/Code/face-up-holdem-ai-engine/training`.
