// Parity corpus (JS side): record reference outputs of the rules core in
// src/engine/. training/scripts/parity_check.py replays the same inputs
// through training/games/holdem.py and asserts identical output.
//
// JS is the source of truth — these rules already ship in production. The
// corpus is regenerated when JS rules change; Python is then re-verified.
//
// Coverage:
//   - handEvaluator: random 7-card hands, recorded (handRank, kickers)
//   - calculatePots / calculateShowdownPots: sampled player configurations
//   - legalActions:  sampled from real hand traces
//   - rules core:    full random-played hands, every state transition logged,
//                    deck recorded so Python can replay with the same shuffle
//   - encode:        sampled mid-hand states encoded under every mover seat
//
// Run from project root:  node training/scripts/parity_corpus.mjs

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  SMALL_BLIND,
  BIG_BLIND,
  STARTING_STACK,
  NUM_SEATS,
  createGame,
  startNewHand,
  applyAction,
  legalActions,
} from '../../src/engine/game.js'
import { evaluateHand, HandRank } from '../../src/engine/handEvaluator.js'
import {
  calculatePots,
  calculateShowdownPots,
} from '../../src/engine/potCalculator.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const OUT_PATH = path.join(here, '..', 'parity_expected.json')

const NUM_TRIALS = 100
const MAX_HANDS_PER_TRIAL = 20
const MAX_ACTIONS_PER_HAND = 500

// Deterministic LCG so the corpus is reproducible across runs.
function makePrng(seed) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 0x100000000
  }
}

const prng = makePrng(1)

const SUITS = ['h', 'd', 'c', 's']

function freshDeck() {
  const cards = []
  for (const suit of SUITS) {
    for (let rank = 2; rank <= 14; rank++) {
      cards.push({ rank, suit })
    }
  }
  return cards
}

function shuffleWith(cards, rng) {
  const a = cards.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

// ── handEvaluator ──────────────────────────────────────────────────

function handEvalCorpus() {
  const cases = []
  const local = makePrng(42)
  for (let i = 0; i < 500; i++) {
    const deck = shuffleWith(freshDeck(), local)
    const seven = deck.slice(0, 7)
    cases.push({ cards: seven, expected: evaluateHand(seven) })
  }
  return cases
}

// ── potCalculator ──────────────────────────────────────────────────

function potCalcCorpus() {
  const cases = []
  const local = makePrng(123)
  for (let trial = 0; trial < 200; trial++) {
    const players = []
    for (let seat = 0; seat < NUM_SEATS; seat++) {
      const r = local()
      let status = 'active'
      if (r < 0.15) status = 'eliminated'
      else if (r < 0.4) status = 'folded'
      else if (r < 0.5) status = 'all_in'
      const currentBet = Math.floor(local() * 50)
      const totalBetThisHand = currentBet + Math.floor(local() * 100)
      players.push({
        seat,
        status,
        currentBet,
        totalBetThisHand,
      })
    }
    cases.push({
      players,
      expectedPots: calculatePots(players),
      expectedShowdownPots: calculateShowdownPots(players),
    })
  }
  return cases
}

// ── Full rules-core traces ─────────────────────────────────────────
//
// Each trial: start a fresh game, then play up to MAX_HANDS_PER_TRIAL
// hands. For each hand, the shuffled deck is recorded immediately after
// startNewHand so Python can reconstruct via start_new_hand(deck=...).

function snapshotState(s) {
  // Shape mirrors what Python produces. Deck is recorded once per hand
  // (post-startNewHand) rather than at every step — its evolution is
  // implicit in communityCards.
  return {
    handNumber: s.handNumber,
    phase: s.phase,
    communityCards: s.communityCards.map((c) => ({ rank: c.rank, suit: c.suit, id: c.id })),
    players: s.players.map((p) => ({
      seat: p.seat,
      name: p.name,
      isHuman: p.isHuman,
      chips: p.chips,
      status: p.status,
      holeCards: p.holeCards.map((c) => ({ rank: c.rank, suit: c.suit, id: c.id })),
      currentBet: p.currentBet,
      totalBetThisHand: p.totalBetThisHand,
      hasActedThisStreet: p.hasActedThisStreet,
      isDealer: p.isDealer,
      isSB: p.isSB,
      isBB: p.isBB,
      handRank: p.handRank ?? null,
      handKickers: p.handKickers ?? null,
    })),
    pots: s.pots.map((p) => ({ amount: p.amount, eligibleSeats: p.eligibleSeats })),
    totalPot: s.totalPot,
    dealerSeat: s.dealerSeat,
    actionOnSeat: s.actionOnSeat,
    lastAggressorSeat: s.lastAggressorSeat,
    currentStreetBet: s.currentStreetBet,
    lastRaiseSize: s.lastRaiseSize,
    callAmount: s.callAmount,
    canCheck: s.canCheck,
    minRaiseAmount: s.minRaiseAmount,
    maxRaiseAmount: s.maxRaiseAmount,
    winners: s.winners.map((w) => ({ seat: w.seat, amountWon: w.amountWon })),
  }
}

function snapshotDeck(deck) {
  return deck.map((c) => ({ rank: c.rank, suit: c.suit, id: c.id }))
}

function pickAction(state) {
  const legal = legalActions(state)
  const choice = { ...legal[Math.floor(prng() * legal.length)] }
  if (choice.type === 'raise') {
    const lo = state.minRaiseAmount
    const hi = state.maxRaiseAmount
    choice.amount = lo + Math.floor(prng() * (hi - lo + 1))
  }
  return choice
}

function rulesTrials() {
  const trials = []
  for (let t = 0; t < NUM_TRIALS; t++) {
    let state = createGame()
    const trial = { trialIndex: t, initialDealer: state.dealerSeat, hands: [] }

    for (let h = 0; h < MAX_HANDS_PER_TRIAL; h++) {
      if (state.phase === 'game_over') break
      const preStartState = snapshotState(state)
      state = startNewHand(state)
      // Reconstruct the pre-deal shuffled deck: walk players in iteration
      // order, push the hole cards they were dealt, then append the
      // remaining deck. This matches what Python's start_new_hand will
      // consume when given `deck=...`.
      const preDealDeck = []
      for (const p of state.players) {
        if (p.status !== 'eliminated') {
          preDealDeck.push(...p.holeCards)
        }
      }
      preDealDeck.push(...state.deck)
      const deck = snapshotDeck(preDealDeck)
      const postStartState = snapshotState(state)
      const transitions = []

      let actions = 0
      while (
        state.phase !== 'hand_complete' &&
        state.phase !== 'game_over' &&
        actions < MAX_ACTIONS_PER_HAND
      ) {
        const action = pickAction(state)
        state = applyAction(state, action)
        transitions.push({ action, afterState: snapshotState(state) })
        actions++
      }

      trial.hands.push({
        handIndex: h,
        preStartState,
        deck,
        postStartState,
        transitions,
      })
    }

    trials.push(trial)
  }
  return trials
}

// ── legalActions corpus (subset of trial states) ────────────────────

function legalActionsCorpus(trials) {
  const cases = []
  for (const trial of trials) {
    for (const hand of trial.hands) {
      // postStartState always has an actionOnSeat (or it would have gone
      // straight to showdown, in which case the phase is already showdown).
      if (hand.postStartState.actionOnSeat >= 0) {
        cases.push({
          state: hand.postStartState,
          expected: legalActionsFromSnapshot(hand.postStartState),
        })
      }
      for (const tr of hand.transitions) {
        if (tr.afterState.actionOnSeat >= 0) {
          cases.push({
            state: tr.afterState,
            expected: legalActionsFromSnapshot(tr.afterState),
          })
        }
      }
      if (cases.length >= 1000) break
    }
    if (cases.length >= 1000) break
  }
  return cases
}

// Compute legalActions from a snapshot. legalActions reads only the
// snapshot-recorded fields, so this is safe.
function legalActionsFromSnapshot(snap) {
  return legalActions(snap)
}

// ── encode (separate file — runs after game.js fields are validated) ──

// Phase-1 encoder parity is asserted by parity_check.py via the Python
// `encode()` in games/holdem.py. The JS-side encoder lives in
// src/engine/nnGame.js, which doesn't exist yet (built in Phase 5). We
// skip the encode corpus here and add it when nnGame.js exists.

// ── Write ──────────────────────────────────────────────────────────

const trials = rulesTrials()
const payload = {
  meta: {
    generator: 'parity_corpus.mjs',
    suits: SUITS,
    handRank: HandRank,
    smallBlind: SMALL_BLIND,
    bigBlind: BIG_BLIND,
    startingStack: STARTING_STACK,
    numSeats: NUM_SEATS,
    numTrials: NUM_TRIALS,
    maxHandsPerTrial: MAX_HANDS_PER_TRIAL,
    seed: 1,
  },
  handEvaluator: handEvalCorpus(),
  potCalculator: potCalcCorpus(),
  trials,
  legalActions: legalActionsCorpus(trials),
}

fs.writeFileSync(OUT_PATH, JSON.stringify(payload))
const totalHands = trials.reduce((a, t) => a + t.hands.length, 0)
const totalTransitions = trials.reduce(
  (a, t) => a + t.hands.reduce((b, h) => b + h.transitions.length, 0),
  0,
)
console.log(
  `wrote parity corpus -> ${OUT_PATH}\n` +
  `  handEvaluator     : ${payload.handEvaluator.length} cases\n` +
  `  potCalculator     : ${payload.potCalculator.length} cases\n` +
  `  trials            : ${trials.length} (${totalHands} hands, ${totalTransitions} transitions)\n` +
  `  legalActions      : ${payload.legalActions.length} cases`,
)
