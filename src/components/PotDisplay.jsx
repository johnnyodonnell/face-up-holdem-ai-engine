export default function PotDisplay({ pots, totalPot }) {
  if (totalPot === 0) return null
  return (
    <div className="pot-display">
      <div className="pot-total">Pot: ${totalPot}</div>
      {pots && pots.length > 1 && (
        <div className="side-pots">
          {pots.map((pot, i) => (
            <span key={i} className="side-pot">
              {i === 0 ? 'Main' : `Side ${i}`}: ${pot.amount}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
