// Pure 7-card Texas Hold'em hand evaluator. No React.
// Ported from personal-poker-trainer/frontend/src/engine/handEvaluator.ts.
// Returns { handRank, kickers } — no description text (we don't display it).

export const HandRank = {
  HIGH_CARD: 0,
  PAIR: 1,
  TWO_PAIR: 2,
  THREE_OF_A_KIND: 3,
  STRAIGHT: 4,
  FLUSH: 5,
  FULL_HOUSE: 6,
  FOUR_OF_A_KIND: 7,
  STRAIGHT_FLUSH: 8,
  ROYAL_FLUSH: 9,
}

export function evaluateHand(sevenCards) {
  let bestRank = null
  let bestKickers = null

  for (const combo of combinations(sevenCards, 5)) {
    const { handRank, kickers } = evaluateFive(combo)
    if (
      bestRank === null ||
      compareHands(handRank, kickers, bestRank, bestKickers) > 0
    ) {
      bestRank = handRank
      bestKickers = kickers
    }
  }

  return { handRank: bestRank, kickers: bestKickers }
}

export function compareHandResults(a, b) {
  return compareHands(a.handRank, a.kickers, b.handRank, b.kickers)
}

function compareHands(rankA, kickersA, rankB, kickersB) {
  if (rankA !== rankB) return rankA - rankB
  for (let i = 0; i < Math.min(kickersA.length, kickersB.length); i++) {
    if (kickersA[i] !== kickersB[i]) return kickersA[i] - kickersB[i]
  }
  return 0
}

function evaluateFive(cards) {
  const ranks = cards.map((c) => c.rank).sort((a, b) => b - a)
  const suits = cards.map((c) => c.suit)
  const isFlush = new Set(suits).size === 1
  const { isStraight, highCard } = checkStraight(ranks)

  const rankCounts = new Map()
  for (const r of ranks) {
    rankCounts.set(r, (rankCounts.get(r) || 0) + 1)
  }

  const groups = Array.from(rankCounts.entries()).sort(
    (a, b) => b[1] - a[1] || b[0] - a[0],
  )
  const counts = groups.map((g) => g[1])

  if (isStraight && isFlush) {
    if (highCard === 14) {
      return { handRank: HandRank.ROYAL_FLUSH, kickers: [14] }
    }
    return { handRank: HandRank.STRAIGHT_FLUSH, kickers: [highCard] }
  }

  if (counts[0] === 4) {
    return {
      handRank: HandRank.FOUR_OF_A_KIND,
      kickers: [groups[0][0], groups[1][0]],
    }
  }

  if (counts[0] === 3 && counts[1] === 2) {
    return {
      handRank: HandRank.FULL_HOUSE,
      kickers: [groups[0][0], groups[1][0]],
    }
  }

  if (isFlush) {
    return { handRank: HandRank.FLUSH, kickers: ranks }
  }

  if (isStraight) {
    return { handRank: HandRank.STRAIGHT, kickers: [highCard] }
  }

  if (counts[0] === 3) {
    const kickers = groups.slice(1).map((g) => g[0]).sort((a, b) => b - a)
    return {
      handRank: HandRank.THREE_OF_A_KIND,
      kickers: [groups[0][0], ...kickers],
    }
  }

  if (counts[0] === 2 && counts[1] === 2) {
    const pairs = [groups[0][0], groups[1][0]].sort((a, b) => b - a)
    return {
      handRank: HandRank.TWO_PAIR,
      kickers: [...pairs, groups[2][0]],
    }
  }

  if (counts[0] === 2) {
    const kickers = groups.slice(1).map((g) => g[0]).sort((a, b) => b - a)
    return {
      handRank: HandRank.PAIR,
      kickers: [groups[0][0], ...kickers],
    }
  }

  return { handRank: HandRank.HIGH_CARD, kickers: ranks }
}

function checkStraight(sortedRanks) {
  const unique = [...new Set(sortedRanks)].sort((a, b) => b - a)
  if (unique.length !== 5) return { isStraight: false, highCard: 0 }
  if (unique[0] - unique[4] === 4) {
    return { isStraight: true, highCard: unique[0] }
  }
  // Wheel: A-2-3-4-5
  if (
    unique[0] === 14 &&
    unique[1] === 5 &&
    unique[2] === 4 &&
    unique[3] === 3 &&
    unique[4] === 2
  ) {
    return { isStraight: true, highCard: 5 }
  }
  return { isStraight: false, highCard: 0 }
}

function* combinations(arr, k) {
  const n = arr.length
  if (k > n) return
  const idx = Array.from({ length: k }, (_, i) => i)
  while (true) {
    yield idx.map((i) => arr[i])
    let i = k - 1
    while (i >= 0 && idx[i] === n - k + i) i--
    if (i < 0) return
    idx[i]++
    for (let j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1
  }
}
