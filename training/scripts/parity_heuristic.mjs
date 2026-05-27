// Heuristic parity corpus (JS side).
//
// Play random hands, sample mid-hand states, record what the JS heuristic
// computes for each (equity value + chosen action index + concrete action).
// `parity_heuristic.py` replays each state through the Python heuristic
// with the same seed and asserts byte-exact agreement.
//
// JS is authoritative because the production engine ships in JS — the
// heuristic doesn't, but using the same JS-source-of-truth convention
// keeps the project consistent.

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  createGame,
  startNewHand,
  applyAction,
  legalActions,
} from '../../src/engine/game.js'
import {
  bestMove,
  equity,
  chooseActionIdx,
  actionForIdx,
  makeLCG,
  DEFAULT_NUM_SAMPLES,
} from '../../src/engine/heuristic.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const OUT_PATH = path.join(here, '..', 'parity_heuristic_expected.json')

const NUM_TRIALS = 30
const MAX_HANDS_PER_TRIAL = 6
const MAX_ACTIONS_PER_HAND = 200
const SAMPLES_PER_DECISION = 80 // smaller than DEFAULT_NUM_SAMPLES for speed
const CORPUS_TARGET = 200       // how many heuristic-decision cases to record

// Deterministic action picker for the *driving* random play that produces
// the states we then evaluate. The heuristic itself runs with its own LCG
// driven by the per-decision seed recorded in the corpus.
function makeDriverPrng(seed) {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 0x100000000
  }
}

const driver = makeDriverPrng(31)

function pickRandomAction(state) {
  const legal = legalActions(state)
  const choice = { ...legal[Math.floor(driver() * legal.length)] }
  if (choice.type === 'raise') {
    const lo = state.minRaiseAmount
    const hi = state.maxRaiseAmount
    choice.amount = lo + Math.floor(driver() * (hi - lo + 1))
  }
  return choice
}

function snapshotState(s) {
  // Subset of fields the heuristic actually reads. Other fields are
  // irrelevant to parity, but we serialize the full state shape that
  // matches the Python rules port so calls into Python's heuristic
  // accept it unchanged.
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

const cases = []
let stateCounter = 0

outer: for (let t = 0; t < NUM_TRIALS && cases.length < CORPUS_TARGET; t++) {
  let state = createGame()
  for (let h = 0; h < MAX_HANDS_PER_TRIAL && state.phase !== 'game_over'; h++) {
    state = startNewHand(state)
    let actions = 0
    while (
      state.phase !== 'hand_complete' &&
      state.phase !== 'game_over' &&
      actions < MAX_ACTIONS_PER_HAND
    ) {
      stateCounter++
      // Record a heuristic case from this state, then drive forward with
      // a random action so the driving trajectory stays unbiased.
      const snapshot = snapshotState(state)
      const seat = state.actionOnSeat
      const seed = ((t * 1000003) ^ (h * 17) ^ (actions * 7919) ^ stateCounter) >>> 0
      const rng = makeLCG(seed)
      const eq = equity(snapshot, seat, SAMPLES_PER_DECISION, rng)
      const idx = chooseActionIdx(snapshot, eq)
      const concrete = actionForIdx(snapshot, idx)
      cases.push({
        state: snapshot,
        seat,
        seed,
        numSamples: SAMPLES_PER_DECISION,
        expectedEquity: eq,
        expectedActionIdx: idx,
        expectedAction: concrete,
      })
      if (cases.length >= CORPUS_TARGET) break outer
      const drv = pickRandomAction(state)
      state = applyAction(state, drv)
      actions++
    }
  }
}

const payload = {
  meta: {
    generator: 'parity_heuristic.mjs',
    numTrials: NUM_TRIALS,
    samplesPerDecision: SAMPLES_PER_DECISION,
    corpusTarget: CORPUS_TARGET,
    defaultNumSamples: DEFAULT_NUM_SAMPLES,
  },
  cases,
}
fs.writeFileSync(OUT_PATH, JSON.stringify(payload))
console.log(`wrote heuristic corpus -> ${OUT_PATH}: ${cases.length} cases`)
