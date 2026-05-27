# Face-Up Texas Hold'em — AI Engine

A browser version of **Face-Up Texas Hold'em**: standard No-Limit Hold'em
rules with one difference — every player's hole cards are visible to all
players at all times. The full-information format makes it useful as a
training tool for building intuition about hand strength, board texture,
and opponent decisions.

The v1 AI engine picks a uniform-random legal action on every turn — it is
intentionally not skillful. The engine exposes a single `bestMove(state)`
function so smarter engines (search, neural net, remote service) can
replace it later without touching the UI.

## Running locally

```sh
npm install
npm run dev        # or: ./run-local.sh
```

Then open the printed URL. `npm run build` produces a production build in `dist/`.

## Rules

- **Deck**: standard 52 cards.
- **Table**: 1 human + 7 bots = 8 seats.
- **Blinds**: $1 / $2 (no-limit).
- **Starting stack**: $200 per player.
- **Hole cards**: dealt face-up — every seat's two cards are always visible.
- **Streets**: preflop → flop → turn → river → showdown, with a betting round on each.
- **Showdown**: best 5-card hand from the 7 available (2 hole + 5 community).
- **Side pots**: built correctly when multiple players go all-in at different totals.
- **Persistent stacks**: chips carry between hands. Game ends when the human busts or all bots are eliminated.

## Project layout

```
src/
  main.jsx                entry — mounts <App>
  App.jsx                 game state + turn flow
  styles/app.css          single global stylesheet
  engine/
    game.js               pure Hold'em rules — no React, no side effects
    handEvaluator.js      7-card best 5-card hand evaluation
    potCalculator.js      main + side pot construction
    random.js             v1 AI — bestMove(state) returns a random legal action
  components/
    Card.jsx              one card (rank + suit, red/black)
    Hand.jsx              two hole cards
    CommunityCards.jsx    flop / turn / river
    PlayerSeat.jsx        name, stack, badges, hole cards, current bet
    PotDisplay.jsx        current pot total
    ActionPanel.jsx       fold / check-or-call / raise (slider + presets)
    Status.jsx            phase + whose turn
    HandResult.jsx        end-of-hand dialog with winners + "Next hand"
    GameOverDialog.jsx    end-of-game dialog with "New game"
```

## Swapping the engine

`src/App.jsx` imports `bestMove` from exactly one engine file. To plug in a
different engine, change that one line:

```js
import { bestMove } from './engine/random.js'   // current
// import { bestMove } from './engine/<other>.js'
```

The engine contract:

```js
bestMove(state)  // state: full GameState (see engine/game.js)
                 // returns: { type: 'fold' | 'check' | 'call' | 'raise', amount? }
                 // Caller invokes only when it's a bot's turn.
```
