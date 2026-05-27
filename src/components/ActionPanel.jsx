import { useEffect, useState } from 'react'

export default function ActionPanel({
  phase,
  humanCanAct,
  canCheck,
  callAmount,
  minRaiseAmount,
  maxRaiseAmount,
  potForSizing,
  currentStreetBet,
  humanCurrentBet,
  onAction,
  onNextHand,
  onNewGame,
}) {
  const [raiseTo, setRaiseTo] = useState(minRaiseAmount)

  useEffect(() => {
    setRaiseTo(minRaiseAmount)
  }, [minRaiseAmount])

  if (phase === 'game_over') {
    return (
      <div className="action-panel">
        <button className="btn btn-deal" onClick={onNewGame}>
          New Game
        </button>
      </div>
    )
  }

  if (phase === 'hand_complete') {
    return (
      <div className="action-panel">
        <button className="btn btn-deal" onClick={onNextHand}>
          Next Hand
        </button>
      </div>
    )
  }

  if (!humanCanAct) return null

  const canRaise = maxRaiseAmount > 0 && maxRaiseAmount > currentStreetBet
  const showSlider = canRaise && minRaiseAmount < maxRaiseAmount

  function clampRaise(amount) {
    return Math.max(minRaiseAmount, Math.min(maxRaiseAmount, Math.round(amount)))
  }
  function presetRaise(fraction) {
    const callAmt = currentStreetBet - humanCurrentBet
    const potAfterCall = potForSizing + callAmt
    return clampRaise(currentStreetBet + callAmt + potAfterCall * fraction)
  }

  return (
    <div className="action-panel">
      {canCheck ? (
        <button
          className="btn btn-check"
          onClick={() => onAction({ type: 'check' })}
        >
          Check
        </button>
      ) : (
        <>
          <button
            className="btn btn-fold"
            onClick={() => onAction({ type: 'fold' })}
          >
            Fold
          </button>
          <button
            className="btn btn-call"
            onClick={() => onAction({ type: 'call' })}
          >
            Call ${callAmount}
          </button>
        </>
      )}

      {canRaise && (
        <div className="raise-controls">
          {showSlider ? (
            <>
              <button
                className="btn btn-preset"
                onClick={() => setRaiseTo(presetRaise(0.5))}
              >
                ½ pot
              </button>
              <button
                className="btn btn-preset"
                onClick={() => setRaiseTo(presetRaise(1))}
              >
                Pot
              </button>
              <button
                className="btn btn-preset"
                onClick={() => setRaiseTo(maxRaiseAmount)}
              >
                All-in
              </button>
              <input
                type="range"
                className="raise-slider"
                min={minRaiseAmount}
                max={maxRaiseAmount}
                value={raiseTo}
                onChange={(e) => setRaiseTo(Number(e.target.value))}
              />
              <button
                className="btn btn-raise"
                onClick={() => onAction({ type: 'raise', amount: raiseTo })}
              >
                Raise to ${raiseTo}
              </button>
            </>
          ) : (
            <button
              className="btn btn-raise"
              onClick={() =>
                onAction({ type: 'raise', amount: maxRaiseAmount })
              }
            >
              All-in ${maxRaiseAmount}
            </button>
          )}
        </div>
      )}
    </div>
  )
}
