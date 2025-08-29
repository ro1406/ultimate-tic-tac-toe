# Ultimate Tic-Tac-Toe with Minimax AI

A Python implementation of Ultimate Tic-Tac-Toe with an AI opponent using the minimax algorithm and tree visualization.

## Game Rules

Ultimate Tic-Tac-Toe is played on a 3×3 grid of 3×3 tic-tac-toe boards. Players take turns placing their mark (X or O) in any empty cell of any board. The location of your move determines which board your opponent must play in next.

### Key Rules:
1. The first player (X) can play anywhere on any board
2. Each subsequent move must be played in the board corresponding to the position of the previous move
3. If a board is won or full, the next player can choose any available board
4. Win the game by getting three of your marks in a row on the large board (horizontally, vertically, or diagonally)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the game:
```bash
python ultimate_tic_tac_toe.py
```

## How to Play

1. **Choose AI Difficulty**: Enter a depth between 1-6 when prompted
   - Depth 1-2: Easy (AI makes basic moves)
   - Depth 3-4: Medium (AI thinks ahead a few moves)
   - Depth 5-6: Hard (AI plays very well)
   - Depth 6+: Perfect play (AI plays optimally)

2. **Making Moves**: Enter moves in the format `board_row, board_col, cell_row, cell_col`
   - All coordinates are 0-indexed (0, 1, or 2)
   - Example: `1, 2, 0, 1` means:
     - Board at row 1, column 2
     - Cell at row 0, column 1 within that board

3. **Game Flow**:
   - You play as X (first player)
   - AI plays as O (second player)
   - The game shows which board you must play in next
   - AI will show its thinking process with a tree visualization

## Features

- **Minimax Algorithm**: AI uses minimax with alpha-beta pruning for optimal play
- **Tree Visualization**: See the AI's decision tree using matplotlib and networkx
- **Difficulty Levels**: Choose how deep the AI searches (1-6+)
- **Console Interface**: Clean ASCII board display
- **Move Validation**: Ensures all moves follow game rules

## Perfect Play Information

For perfect play in Ultimate Tic-Tac-Toe:
- **Maximum practical depth**: 6-8 moves ahead
- **Optimal depth**: 6+ (though this can be computationally expensive)
- **Game complexity**: Much higher than regular tic-tac-toe due to branching factor

## Controls

- Enter moves as: `board_row, board_col, cell_row, cell_col`
- Type `quit` to exit the game
- Use Ctrl+C to interrupt the game

## Example Game

```
==================================================
ULTIMATE TIC-TAC-TOE
==================================================
Format: board_row, board_col, cell_row, cell_col
Example: 1, 2, 0, 1 means board at row 1, col 2, cell at row 0, col 1
==================================================
  -----------------------------------
  |   |   |   | |   |   |   | |   |   |   |
  |   |   |   | |   |   |   | |   |   |   |
  |   |   |   | |   |   |   | |   |   |   |
  -----------------------------------
  |   |   |   | |   |   |   | |   |   |   |
  |   |   |   | |   |   |   | |   |   |   |
  |   |   |   | |   |   |   | |   |   |   |
  -----------------------------------
  |   |   |   | |   |   |   | |   |   |   |
  |   |   |   | |   |   |   | |   |   |   |
  |   |   |   | |   |   |   | |   |   |   |
  -----------------------------------
Next move can be in any board
==================================================
```

## Technical Details

- **AI Algorithm**: Minimax with alpha-beta pruning
- **Evaluation Function**: Considers won boards and positional advantages
- **Visualization**: NetworkX graph with matplotlib rendering
- **Performance**: Optimized for reasonable response times up to depth 6

## Dependencies

- `matplotlib`: For tree visualization
- `networkx`: For graph representation
- `copy`: For deep copying game states
- `time`: For measuring AI thinking time
