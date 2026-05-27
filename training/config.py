"""Central configuration for the Face-Up Hold'em AlphaZero pipeline.

Every tunable lives here. Values the browser engine also needs (action size,
input size, hidden size, trunk layers) are re-emitted into weights.json by
scripts/export_weights.py so the JS side stays in sync.
"""

# -- Reproducibility -------------------------------------------------------
SEED = 1234

# -- Network ---------------------------------------------------------------
HIDDEN_SIZE = 256
TRUNK_LAYERS = 3          # number of Linear+ReLU layers before the heads

# -- MCTS ------------------------------------------------------------------
C_PUCT = 1.5
# Training-time MCTS budget per decision.
NUM_SIMULATIONS = 80
# Number of deck samples per chance-node expansion (flop/turn/river).
CHANCE_SAMPLES = 4

# Search budget used at evaluation time. v1 ships argmax-policy in the browser
# (no MCTS), so PLAY_SIMULATIONS only applies to the Python-side evaluator.
PLAY_SIMULATIONS = 160

DIRICHLET_ALPHA = 0.6     # 7 actions; alpha tuned higher than fox-lite (0.5/33)
DIRICHLET_EPSILON = 0.25  # mixing weight (self-play only)

# -- Self-play -------------------------------------------------------------
# T=1 sampling for the first N decisions of a hand, then argmax. A typical
# hand has at most ~30 decisions across 8 actors × 4 streets.
TEMPERATURE_DECISIONS = 30

# -- Training --------------------------------------------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
TRAIN_STEPS_PER_ITER = 200

# -- Replay buffer ---------------------------------------------------------
# Hands produce many decisions (up to ~30); buffer sized accordingly.
BUFFER_SIZE = 500_000

# -- Training loop ---------------------------------------------------------
NUM_ITERATIONS = 40
GAMES_PER_ITER = 64       # hands per iteration

# -- Arena (promotion gate) ------------------------------------------------
# A "match" is N hands where 1 seat plays challenger and 7 play champion.
# Score = challenger's chips_delta / BIG_BLIND, summed across hands, divided
# by hands × 100 → BB/100. Promote if average BB/100 exceeds the threshold.
ARENA_MATCHES = 8         # one per starting seat, rotates challenger's position
ARENA_HANDS_PER_MATCH = 200
ARENA_BB100_THRESHOLD = 3.0

# -- Safety tripwires ------------------------------------------------------
MAX_WEIGHT_NORM = 1e4     # halt training if any param's L2 norm blows past this
