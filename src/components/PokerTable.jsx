// Grid layout copied from personal-poker-trainer/frontend/src/components/PokerTable.tsx.
// 5 columns × 3 rows. Seats placed in the 8 outer slots, community + pot in the
// center span. SEAT_LAYOUT maps player-index → grid-class so seat 0 (the human)
// sits at bottom-middle and indices proceed clockwise around the table.

const SEAT_LAYOUT = [
  { index: 3, gridClass: 'seat-4' },
  { index: 4, gridClass: 'seat-5' },
  { index: 5, gridClass: 'seat-6' },
  { index: 2, gridClass: 'seat-3' },
  { index: 6, gridClass: 'seat-7' },
  { index: 1, gridClass: 'seat-2' },
  { index: 0, gridClass: 'seat-0' },
  { index: 7, gridClass: 'seat-1' },
]

export default function PokerTable({ renderSeat, centerContent }) {
  return (
    <div className="table-felt">
      {SEAT_LAYOUT.map(({ index, gridClass }) => (
        <div key={index} className={`seat-slot ${gridClass}`}>
          {renderSeat(index)}
        </div>
      ))}
      <div className="community-area">{centerContent}</div>
    </div>
  )
}
