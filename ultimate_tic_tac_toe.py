import copy
import time
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque


class UltimateTicTacToe:
    def __init__(self):
        # Initialize the game board: 3x3 grid of 3x3 boards
        # Each board is represented as a 3x3 grid
        # 0 = empty, 1 = X, 2 = O
        self.board = [[[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
        self.current_player = 1  # 1 for X, 2 for O
        self.next_board = None  # None means any board is valid
        self.game_over = False
        self.winner = None

    def print_board(self):
        """Print the current state of the board in ASCII format"""
        print("\n" + "=" * 50)
        print("ULTIMATE TIC-TAC-TOE")
        print("=" * 50)
        print("Format: board_row, board_col, cell_row, cell_col")
        print("Example: 1, 2, 0, 1 means board at row 1, col 2, cell at row 0, col 1")
        print("=" * 50)

        for big_row in range(3):
            for small_row in range(3):
                # Print row separator
                if small_row == 0:
                    print("  " + "-" * 35)

                # Print the actual row
                row_str = "  "
                for big_col in range(3):
                    row_str += "| "
                    for small_col in range(3):
                        cell = self.board[big_row][big_col][small_row][small_col]
                        if cell == 0:
                            row_str += " "
                        elif cell == 1:
                            row_str += "X"
                        else:
                            row_str += "O"
                        if small_col < 2:
                            row_str += " "
                    row_str += " | "
                print(row_str)

            # Print bottom separator for each big board
            print("  " + "-" * 35)

        # Show next board constraint
        if self.next_board is not None:
            print(f"Next move must be in board: ({self.next_board[0]}, {self.next_board[1]})")
        else:
            print("Next move can be in any board")

        print("=" * 50)

    def is_valid_move(self, board_row: int, board_col: int, cell_row: int, cell_col: int) -> bool:
        """Check if a move is valid"""
        # Check if the move is in the correct board
        if self.next_board is not None:
            if (board_row, board_col) != self.next_board:
                return False

        # Check if the board is still playable
        if self.is_board_won(board_row, board_col) or self.is_board_full(board_row, board_col):
            return False

        # Check if the cell is empty
        return self.board[board_row][board_col][cell_row][cell_col] == 0

    def make_move(self, board_row: int, board_col: int, cell_row: int, cell_col: int) -> bool:
        """Make a move and return True if successful"""
        if not self.is_valid_move(board_row, board_col, cell_row, cell_col):
            return False

        # Make the move
        self.board[board_row][board_col][cell_row][cell_col] = self.current_player

        # Set next board constraint
        next_board = (cell_row, cell_col)

        # Check if the next board is already won or full
        if self.is_board_won(next_board[0], next_board[1]) or self.is_board_full(next_board[0], next_board[1]):
            # Next board is not playable, so next player gets free choice
            self.next_board = None
        else:
            # Next board is playable, set the constraint
            self.next_board = next_board

        # Check if the current board is won
        if self.is_board_won(board_row, board_col):
            # Mark the board as won by the current player
            pass  # We'll handle this in the evaluation function

        # Check if the game is over
        if self.is_game_over():
            self.game_over = True
            self.winner = self.get_winner()

        # Switch players
        self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1

        return True

    def is_board_won(self, board_row: int, board_col: int) -> bool:
        """Check if a specific board is won"""
        board = self.board[board_row][board_col]

        # Check rows
        for row in range(3):
            if board[row][0] != 0 and board[row][0] == board[row][1] == board[row][2]:
                return True

        # Check columns
        for col in range(3):
            if board[0][col] != 0 and board[0][col] == board[1][col] == board[2][col]:
                return True

        # Check diagonals
        if board[0][0] != 0 and board[0][0] == board[1][1] == board[2][2]:
            return True
        if board[0][2] != 0 and board[0][2] == board[1][1] == board[2][0]:
            return True

        return False

    def is_board_full(self, board_row: int, board_col: int) -> bool:
        """Check if a specific board is full"""
        board = self.board[board_row][board_col]
        return all(board[i][j] != 0 for i in range(3) for j in range(3))

    def get_board_winner(self, board_row: int, board_col: int) -> Optional[int]:
        """Get the winner of a specific board (1 for X, 2 for O, None if not won)"""
        if not self.is_board_won(board_row, board_col):
            return None

        board = self.board[board_row][board_col]
        # Check rows
        for row in range(3):
            if board[row][0] != 0 and board[row][0] == board[row][1] == board[row][2]:
                return board[row][0]

        # Check columns
        for col in range(3):
            if board[0][col] != 0 and board[0][col] == board[1][col] == board[2][col]:
                return board[0][col]

        # Check diagonals
        if board[0][0] != 0 and board[0][0] == board[1][1] == board[2][2]:
            return board[0][0]
        if board[0][2] != 0 and board[0][2] == board[1][1] == board[2][0]:
            return board[0][2]

        return None

    def is_game_over(self) -> bool:
        """Check if the game is over"""
        # Check if any player has won the big board
        if self.get_winner() is not None:
            return True

        # Check if all boards are full or won
        for i in range(3):
            for j in range(3):
                if not self.is_board_won(i, j) and not self.is_board_full(i, j):
                    return False

        return True

    def get_winner(self) -> Optional[int]:
        """Get the winner of the game (1 for X, 2 for O, None if no winner)"""
        # Create a 3x3 representation of won boards
        big_board = [[0 for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                winner = self.get_board_winner(i, j)
                if winner is not None:
                    big_board[i][j] = winner

        # Check rows
        for row in range(3):
            if big_board[row][0] != 0 and big_board[row][0] == big_board[row][1] == big_board[row][2]:
                return big_board[row][0]

        # Check columns
        for col in range(3):
            if big_board[0][col] != 0 and big_board[0][col] == big_board[1][col] == big_board[2][col]:
                return big_board[0][col]

        # Check diagonals
        if big_board[0][0] != 0 and big_board[0][0] == big_board[1][1] == big_board[2][2]:
            return big_board[0][0]
        if big_board[0][2] != 0 and big_board[0][2] == big_board[1][1] == big_board[2][0]:
            return big_board[0][2]

        return None

    def get_valid_moves(self) -> List[Tuple[int, int, int, int]]:
        """Get all valid moves for the current player"""
        moves = []

        if self.next_board is not None:
            # Must play in the specified board
            board_row, board_col = self.next_board
            # Check if the specified board is still playable
            if not self.is_board_won(board_row, board_col) and not self.is_board_full(board_row, board_col):
                # Board is playable, must play here
                for cell_row in range(3):
                    for cell_col in range(3):
                        if self.board[board_row][board_col][cell_row][cell_col] == 0:
                            moves.append((board_row, board_col, cell_row, cell_col))
            else:
                # Board is won or full, can play in any available board
                for board_row in range(3):
                    for board_col in range(3):
                        if not self.is_board_won(board_row, board_col) and not self.is_board_full(board_row, board_col):
                            for cell_row in range(3):
                                for cell_col in range(3):
                                    if self.board[board_row][board_col][cell_row][cell_col] == 0:
                                        moves.append((board_row, board_col, cell_row, cell_col))
        else:
            # Can play in any board
            for board_row in range(3):
                for board_col in range(3):
                    if not self.is_board_won(board_row, board_col) and not self.is_board_full(board_row, board_col):
                        for cell_row in range(3):
                            for cell_col in range(3):
                                if self.board[board_row][board_col][cell_row][cell_col] == 0:
                                    moves.append((board_row, board_col, cell_row, cell_col))

        return moves

    def evaluate_position(self) -> float:
        """Evaluate the current position for minimax"""
        winner = self.get_winner()
        if winner == 2:  # O wins
            return 1000
        elif winner == 1:  # X wins
            return -1000

        # Count won boards
        o_boards = 0
        x_boards = 0

        for i in range(3):
            for j in range(3):
                board_winner = self.get_board_winner(i, j)
                if board_winner == 2:
                    o_boards += 1
                elif board_winner == 1:
                    x_boards += 1

        # Add positional bonus for boards that are close to winning
        o_positional = 0
        x_positional = 0

        for i in range(3):
            for j in range(3):
                if not self.is_board_won(i, j) and not self.is_board_full(i, j):
                    # Count potential wins for this board
                    o_potential = self.count_potential_wins(i, j, 2)
                    x_potential = self.count_potential_wins(i, j, 1)
                    o_positional += o_potential
                    x_positional += x_potential

        return (o_boards - x_boards) * 10 + (o_positional - x_positional) * 0.1

    def count_potential_wins(self, board_row: int, board_col: int, player: int) -> int:
        """Count potential winning lines for a player in a specific board"""
        board = self.board[board_row][board_col]
        count = 0

        # Check rows
        for row in range(3):
            line = [board[row][0], board[row][1], board[row][2]]
            if self.can_win_line(line, player):
                count += 1

        # Check columns
        for col in range(3):
            line = [board[0][col], board[1][col], board[2][col]]
            if self.can_win_line(line, player):
                count += 1

        # Check diagonals
        diag1 = [board[0][0], board[1][1], board[2][2]]
        if self.can_win_line(diag1, player):
            count += 1

        diag2 = [board[0][2], board[1][1], board[2][0]]
        if self.can_win_line(diag2, player):
            count += 1

        return count

    def can_win_line(self, line: List[int], player: int) -> bool:
        """Check if a player can still win a line"""
        opponent = 3 - player
        return opponent not in line and line.count(player) >= 1

    def minimax(
        self,
        depth: int,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
        maximizing: bool = True,
        tree_data: Optional[Dict] = None,
        node_id: str = "root",
    ) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        """Minimax algorithm with alpha-beta pruning and tree visualization"""
        if tree_data is None:
            tree_data = {"nodes": {}, "edges": []}

        # Base cases
        if depth == 0 or self.is_game_over():
            evaluation = self.evaluate_position()
            tree_data["nodes"][node_id] = {
                "evaluation": evaluation,
                "depth": depth,
                "game_over": self.is_game_over(),
                "winner": self.get_winner(),
            }
            return evaluation, None

        valid_moves = self.get_valid_moves()
        if not valid_moves:
            evaluation = self.evaluate_position()
            tree_data["nodes"][node_id] = {
                "evaluation": evaluation,
                "depth": depth,
                "game_over": True,
                "winner": self.get_winner(),
            }
            return evaluation, None

        best_move = None

        if maximizing:
            max_eval = float("-inf")
            for i, move in enumerate(valid_moves):
                # Create a copy of the game state
                game_copy = copy.deepcopy(self)
                game_copy.make_move(*move)

                child_id = f"{node_id}_{i}"
                tree_data["edges"].append((node_id, child_id, f"O: {move}"))

                eval_score, _ = game_copy.minimax(depth - 1, alpha, beta, False, tree_data, child_id)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break

            tree_data["nodes"][node_id] = {
                "evaluation": max_eval,
                "depth": depth,
                "best_move": best_move,
                "game_over": False,
            }
            return max_eval, best_move
        else:
            min_eval = float("inf")
            for i, move in enumerate(valid_moves):
                # Create a copy of the game state
                game_copy = copy.deepcopy(self)
                game_copy.make_move(*move)

                child_id = f"{node_id}_{i}"
                tree_data["edges"].append((node_id, child_id, f"X: {move}"))

                eval_score, _ = game_copy.minimax(depth - 1, alpha, beta, True, tree_data, child_id)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break

            tree_data["nodes"][node_id] = {
                "evaluation": min_eval,
                "depth": depth,
                "best_move": best_move,
                "game_over": False,
            }
            return min_eval, best_move

    def visualize_minimax_tree(self, tree_data: Dict, max_depth: int = 3):
        """Visualize the minimax tree using matplotlib and networkx with hierarchical layout"""
        if not tree_data["nodes"]:
            print("No tree data to visualize")
            return

        G = nx.DiGraph()

        # Add nodes
        for node_id, node_data in tree_data["nodes"].items():
            if node_data["depth"] <= max_depth:
                label = f"Eval: {node_data['evaluation']:.1f}\nDepth: {node_data['depth']}"
                if "best_move" in node_data and node_data["best_move"]:
                    label += f"\nBest: {node_data['best_move']}"
                G.add_node(node_id, label=label, depth=node_data["depth"])

        # Add edges
        for edge in tree_data["edges"]:
            parent, child, move_label = edge
            if parent in tree_data["nodes"] and child in tree_data["nodes"]:
                if tree_data["nodes"][parent]["depth"] <= max_depth and tree_data["nodes"][child]["depth"] <= max_depth:
                    G.add_edge(parent, child, label=move_label)

        if len(G.nodes()) == 0:
            print("No nodes to visualize within the specified depth")
            return

        # Create hierarchical layout
        plt.figure(figsize=(20, 12))

        # Use hierarchical layout instead of spring layout
        pos = nx.kamada_kawai_layout(G)

        # Organize nodes by depth for better tree structure
        depth_nodes = {}
        for node, data in G.nodes(data=True):
            depth = data.get("depth", 0)
            if depth not in depth_nodes:
                depth_nodes[depth] = []
            depth_nodes[depth].append(node)

        # Create manual hierarchical positioning
        pos = {}
        max_depth_val = max(depth_nodes.keys()) if depth_nodes else 0

        for depth in range(max_depth_val + 1):
            if depth in depth_nodes:
                nodes_at_depth = depth_nodes[depth]
                y_pos = max_depth_val - depth  # Root at top

                for i, node in enumerate(nodes_at_depth):
                    x_pos = (i - len(nodes_at_depth) / 2) * 2  # Center nodes at each level
                    pos[node] = (x_pos, y_pos)

        # Draw nodes with different colors based on depth
        node_colors = []
        for node in G.nodes():
            depth = G.nodes[node].get("depth", 0)
            if depth == 0:
                node_colors.append("lightgreen")  # Root
            elif depth == 1:
                node_colors.append("lightblue")  # Level 1
            elif depth == 2:
                node_colors.append("lightcoral")  # Level 2
            else:
                node_colors.append("lightyellow")  # Level 3+

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, edgecolors="black", linewidths=2)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, arrowstyle="->", width=2)

        # Draw node labels with better formatting
        labels = {node: G.nodes[node]["label"] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight="bold")

        # Draw edge labels with better positioning
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels, font_size=7, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

        plt.title(f"Minimax Tree Visualization (Depth {max_depth})", fontsize=16, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def print_minimax_tree_console(self, tree_data: Dict, max_depth: int = 3):
        """Print a simple console-based tree visualization"""
        if not tree_data["nodes"]:
            print("No tree data to visualize")
            return

        print("\n" + "=" * 60)
        print("MINIMAX TREE VISUALIZATION (CONSOLE)")
        print("=" * 60)

        # Organize nodes by depth
        depth_nodes = {}
        for node_id, node_data in tree_data["nodes"].items():
            if node_data["depth"] <= max_depth:
                depth = node_data["depth"]
                if depth not in depth_nodes:
                    depth_nodes[depth] = []
                depth_nodes[depth].append((node_id, node_data))

        # Print tree in hierarchical format
        for depth in sorted(depth_nodes.keys()):
            print(f"\n--- DEPTH {depth} ---")
            nodes_at_depth = depth_nodes[depth]

            for i, (node_id, node_data) in enumerate(nodes_at_depth):
                indent = "  " * depth
                eval_score = node_data["evaluation"]
                best_move = node_data.get("best_move", "None")

                print(f"{indent}Node {i+1}: Eval={eval_score:.1f}, Best={best_move}")

                # Show children
                children = [edge for edge in tree_data["edges"] if edge[0] == node_id]
                for child_edge in children:
                    child_id = child_edge[1]
                    move_label = child_edge[2]
                    if child_id in tree_data["nodes"] and tree_data["nodes"][child_id]["depth"] <= max_depth:
                        child_eval = tree_data["nodes"][child_id]["evaluation"]
                        print(f"{indent}  └─ {move_label} → Eval={child_eval:.1f}")

        print("\n" + "=" * 60)


def get_user_move(game: UltimateTicTacToe) -> Tuple[int, int, int, int]:
    """Get a valid move from the user"""
    while True:
        try:
            move_input = input("Enter your move (board_row, board_col, cell_row, cell_col): ").strip()
            if move_input.lower() == "quit":
                return None

            parts = [int(x.strip()) for x in move_input.split(",")]
            if len(parts) != 4:
                print("Invalid format. Please use: board_row, board_col, cell_row, cell_col")
                continue

            board_row, board_col, cell_row, cell_col = parts

            if not (0 <= board_row <= 2 and 0 <= board_col <= 2 and 0 <= cell_row <= 2 and 0 <= cell_col <= 2):
                print("All coordinates must be between 0 and 2")
                continue

            if not game.is_valid_move(board_row, board_col, cell_row, cell_col):
                print("Invalid move. Please try again.")
                continue

            return board_row, board_col, cell_row, cell_col

        except ValueError:
            print("Invalid input. Please enter four numbers separated by commas.")
        except KeyboardInterrupt:
            print("\nGame interrupted.")
            return None


def main():
    print("Welcome to Ultimate Tic-Tac-Toe!")
    print("You are X, the AI is O")

    # Get difficulty level
    while True:
        try:
            depth = int(input("Enter AI depth (1-10, higher = stronger): "))
            if 1 <= depth <= 10:
                break
            else:
                print("Depth must be between 1 and 10")
        except ValueError:
            print("Please enter a valid number")

    print(f"AI depth set to {depth}")
    print("Note: For perfect play, use depth 10 or higher")

    # Get visualization preference
    while True:
        viz_choice = input("Choose visualization type (0=Off, 1=Graphical, 2=Console, 3=Both): ").strip()
        if viz_choice in ["1", "2", "3", "0"]:
            break
        else:
            print("Please enter 0, 1, 2, or 3")

    game = UltimateTicTacToe()

    while not game.game_over:
        game.print_board()

        if game.current_player == 1:  # User's turn (X)
            print("Your turn (X)")
            move = get_user_move(game)
            if move is None:
                print("Game ended.")
                break

            game.make_move(*move)
            print(f"You played: {move}")

        else:  # AI's turn (O)
            print("AI is thinking...")
            start_time = time.time()

            # Run minimax with tree visualization
            tree_data = {"nodes": {}, "edges": []}
            evaluation, best_move = game.minimax(depth, tree_data=tree_data)

            end_time = time.time()
            print(f"AI evaluation: {evaluation:.1f}")
            print(f"AI thinking time: {end_time - start_time:.2f} seconds")

            # Visualize the tree based on user preference
            if viz_choice in ["1", "3"]:
                print("Visualizing minimax tree (graphical)...")
                try:
                    game.visualize_minimax_tree(tree_data, min(depth, 3))
                except Exception as e:
                    print(f"Graphical visualization failed: {e}")
                    print("Falling back to console visualization...")
                    game.print_minimax_tree_console(tree_data, min(depth, 3))

            if viz_choice in ["2", "3"]:
                print("Visualizing minimax tree (console)...")
                game.print_minimax_tree_console(tree_data, min(depth, 3))

            if best_move:
                game.make_move(*best_move)
                print(f"AI played: {best_move}")
            else:
                print("AI couldn't find a valid move!")
                break

    # Game over
    game.print_board()
    if game.winner == 1:
        print("Congratulations! You won!")
    elif game.winner == 2:
        print("AI won! Better luck next time!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    main()
