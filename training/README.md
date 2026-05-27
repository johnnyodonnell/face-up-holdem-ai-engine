# Training — neural Face-Up Hold'em bot

This folder trains the project's neural bot via **self-play reinforcement
learning** — no human games, no hand-coded teacher. The bot is an
AlphaZero-style policy/value network guided by PUCT MCTS with chance nodes
for the flop / turn / river deals.

Face-up Hold'em is a **perfect-information stochastic game** — every player's
hole cards are always visible, so the only hidden information is the future
board cards (which are unknown the same way to everyone). That means standard
AlphaZero machinery applies; we do not need CFR-class imperfect-information
techniques, and we do not need fox-lite's PIMC determinization layer.

The web app (`../src/`) ships only the trained weights and a hand-written JS
forward pass — no Python and no ML runtime in the browser.

## Setup (run on `asus-nvidia`)

```sh
ssh asus-nvidia
cd ~/Code/face-up-holdem-ai-engine/training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls PyTorch from the nightly **cu130** wheel index because
GB10 (Blackwell, compute capability `sm_121`) needs CUDA 13. Stable PyTorch
wheels for aarch64 + CUDA 13 don't exist yet.

Verify the GPU is visible:

```sh
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
# -> NVIDIA GB10
```

## Workflow

All commands assume the venv is active and are run from this `training/`
directory.

```sh
python scripts/run_training.py                  # self-play -> train -> arena, looped
python scripts/run_training.py --iterations 2 --games-per-iter 4   # smoke test
python scripts/export_weights.py                # best.pt -> ../src/engine/weights.json
python scripts/evaluate_heuristic.py            # NN vs heuristic, report BB/100
```

`run_training.py` writes `checkpoints/{best,latest}.pt`. Best is replaced
only when a challenger beats the prior best by ≥`ARENA_BB100_THRESHOLD` BB/100
across a rotated-seat tournament, which filters out the occasional silent
regression characteristic of self-play loops.

## Layout

| Path | Role |
| --- | --- |
| `games/hand_evaluator.py` | 7-card best-5 hand evaluation (port of `handEvaluator.js`) |
| `games/holdem.py` | rules + pot calculation + `Game` adapter + state encoder (port of `game.js` + `potCalculator.js`) |
| `game_interface.py` | the N-player `Game` contract the training pipeline depends on |
| `heuristic.py` | equity-based baseline; absolute-strength yardstick only — not shipped |
| `alphazero/network.py` | small policy/value MLP (`PolicyValueNet`) |
| `alphazero/mcts.py` | PUCT MCTS with decision + chance nodes |
| `alphazero/selfplay.py` | one self-play hand → `(input, policy, value)` records |
| `alphazero/replay_buffer.py` | FIFO buffer |
| `alphazero/train.py` | masked-CE policy loss + MSE value loss, NaN guard + weight-norm tripwire |
| `alphazero/arena.py` | BB/100 promotion gate, rotated challenger seat |
| `config.py` | every tunable |
| `scripts/run_training.py` | end-to-end loop |
| `scripts/export_weights.py` | checkpoint → JSON for the browser |
| `scripts/parity_corpus.mjs` + `parity_check.py` | rules-port parity |
| `scripts/network_parity_dump.py` + `network_parity_check.mjs` | forward-pass parity |

## Parity with the browser engine

`src/engine/game.js` is the source of truth for the rules; `games/holdem.py`
is a port. They are kept in lock-step by a deterministic corpus check:

```sh
node training/scripts/parity_corpus.mjs   # JS dumps reference outputs
python training/scripts/parity_check.py   # Python replays and asserts
```

The corpus covers a broad sample of `legalActions`, `applyAction` transitions,
and full random-played hands — every state transition through the rules core
is checked against the JS reference. We deliberately do not try to share an
RNG between languages; the corpus records JS-side outcomes and Python asserts
identical transitions on identical inputs.

The corpus file (`training/parity_expected.json`) is generated, not committed.

For the network itself, a second pair of scripts runs the forward pass in
float64 on both sides and asserts agreement to ~1e-12:

```sh
python scripts/network_parity_dump.py     # Python forward on fixed inputs
node   scripts/network_parity_check.mjs   # JS forward, compare
```
