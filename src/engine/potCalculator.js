// Ported from personal-poker-trainer/frontend/src/engine/potCalculator.ts.
// Builds main + side pots from per-player bet contributions.

export function calculatePots(players) {
  return buildPots(players, (p) => p.currentBet)
}

export function calculateShowdownPots(players) {
  return buildPots(players, (p) => p.totalBetThisHand)
}

function buildPots(players, getBet) {
  const activeBets = []
  let foldedBetsTotal = 0

  for (const p of players) {
    if (p.status === 'eliminated') continue
    const bet = getBet(p)
    if (bet === 0) continue
    if (p.status === 'folded') {
      foldedBetsTotal += bet
    } else {
      activeBets.push({ seat: p.seat, bet })
    }
  }

  if (activeBets.length === 0) return []

  activeBets.sort((a, b) => a.bet - b.bet)

  const uniqueLevels = [...new Set(activeBets.map((b) => b.bet))].sort(
    (a, b) => a - b,
  )

  const pots = []
  let previousLevel = 0
  let remaining = [...activeBets]

  for (const level of uniqueLevels) {
    const increment = level - previousLevel
    if (increment > 0) {
      const potAmount = increment * remaining.length
      const eligibleSeats = remaining.map((b) => b.seat)
      pots.push({ amount: potAmount, eligibleSeats })
    }
    remaining = remaining.filter((b) => b.bet > level)
    previousLevel = level
  }

  if (pots.length > 0 && foldedBetsTotal > 0) {
    pots[0].amount += foldedBetsTotal
  }

  return pots
}

export function getTotalPot(pots) {
  return pots.reduce((sum, pot) => sum + pot.amount, 0)
}
