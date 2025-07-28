import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os
import signal
import sys
import random
import asyncio
import time

# Neural Network for policy and value (matches previous architecture)
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.policy_head = nn.Linear(256, 4096)  # Max possible moves
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, x):
        x = x.view(-1, 12, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value

# MCTS Node
class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0
        self.legal_moves = list(board.legal_moves)

# MCTS Implementation
class MCTS:
    def __init__(self, model, c_puct=1.0, num_simulations=100):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
    
    def board_to_tensor(self, board):
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                color_offset = 6 if piece.color == chess.BLACK else 0
                tensor[piece_map[piece.piece_type] + color_offset][7-rank][file] = 1
        return torch.tensor(tensor, dtype=torch.float32)
    
    def get_policy(self, board):
        tensor = self.board_to_tensor(board)
        with torch.no_grad():
            policy_logits, _ = self.model(tensor.unsqueeze(0))
        policy = torch.softmax(policy_logits, dim=1).squeeze().numpy()
        return policy
    
    def simulate(self, node):
        if node.board.is_game_over():
            result = node.board.result()
            if result == "1-0":
                return 1.0 if node.board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                return -1.0 if node.board.turn == chess.BLACK else 1.0
            return 0.0
        
        if not node.children:
            policy = self.get_policy(node.board)
            tensor = self.board_to_tensor(node.board)
            with torch.no_grad():
                _, value = self.model(tensor.unsqueeze(0))
            value = value.item()
            
            for move in node.legal_moves:
                child_board = node.board.copy()
                child_board.push(move)
                child = MCTSNode(child_board, node, move)
                move_idx = self.move_to_index(move)
                child.prior = policy[move_idx] if move_idx < len(policy) else 0.0
                node.children.append(child)
            return value
        
        best_child = max(node.children, key=lambda c: c.value/c.visits + self.c_puct * c.prior * (np.sqrt(node.visits)/(1+c.visits)) if c.visits > 0 else float('inf'))
        value = -self.simulate(best_child)
        best_child.visits += 1
        best_child.value += value
        node.visits += 1
        node.value += value
        return value
    
    def search(self, board):
        root = MCTSNode(board)
        for _ in range(self.num_simulations):
            self.simulate(root)
        
        move_probs = {}
        total_visits = max(root.visits, 1)
        for child in root.children:
            move_probs[child.move.uci()] = child.visits / total_visits
        if move_probs:
            prob_sum = sum(move_probs.values())
            if prob_sum > 0:
                for move in move_probs:
                    move_probs[move] /= prob_sum
            else:
                n_moves = len(move_probs)
                for move in move_probs:
                    move_probs[move] = 1.0 / n_moves
        return move_probs, root
    
    def move_to_index(self, move):
        from_sq = move.from_square
        to_sq = move.to_square
        return from_sq * 64 + to_sq

# Stockfish API
def get_stockfish_best_move(fen: str, depth: int = 15):
    url = "https://stockfish.online/api/s/v2.php"
    params = {"fen": fen, "depth": depth}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json()
        if result.get("success"):
            return result.get("bestmove")
        else:
            print("API Error:", result.get("error"))
            return None
    except requests.RequestException as e:
        print("HTTP Error:", e)
        return None

def extract_best_move(raw_bestmove: str) -> str:
    if raw_bestmove and raw_bestmove.startswith("bestmove"):
        return raw_bestmove.split()[1]
    return raw_bestmove.strip() if raw_bestmove else None

# Material Points Evaluation
def evaluate_material(board):
    points = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    white_points = 0
    black_points = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = points[piece.piece_type]
            if piece.color == chess.WHITE:
                white_points += value
            else:
                black_points += value
    return white_points, black_points

# Model Management
MODEL_FILE = "chess_model.pth"
model = ChessNet()

def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
            model.eval()  # Set to evaluation mode
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Exiting as model is required.")
            sys.exit(1)
    else:
        print(f"Model file {MODEL_FILE} not found. Please train the model first.")
        sys.exit(1)

# Ctrl+C handler (no saving)
def signal_handler(sig, frame):
    print("\nCtrl+C detected. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

async def play_game(game_number):
    board = chess.Board()
    mcts = MCTS(model, num_simulations=100)
    move_count = 0
    start_time = time.time()
    
    print(f"\nGame {game_number + 1} - Starting FEN: {board.fen()}")
    
    while not board.is_game_over():
        fen_parts = board.fen().split()
        fullmove_number = int(fen_parts[5])
        termination_reason = None
        if fullmove_number >= 50:
            white_points, black_points = evaluate_material(board)
            if white_points > black_points:
                result = "1-0"
                termination_reason = "100-move rule (White material advantage)"
            elif black_points > white_points:
                result = "0-1"
                termination_reason = "100-move rule (Black material advantage)"
            else:
                result = "1/2-1/2"
                termination_reason = "100-move rule (equal material)"
            print(f"Game {game_number + 1} ended due to 100-move rule. Material: White={white_points}, Black={black_points}")
            break
        
        print(f"\nGame {game_number + 1} - Current position:")
        print(board)
        print(f"FEN: {board.fen()}")
        white_points, black_points = evaluate_material(board)
        print(f"Material points: White={white_points}, Black={black_points}")
        
        if board.turn == chess.WHITE:
            player = "White (Neural Network)"
            print(f"{player}'s turn to move...")
            move_probs, _ = mcts.search(board)
            moves = list(move_probs.keys())
            probs = list(move_probs.values())
            if not moves:
                print("No valid moves found.")
                result = "0-1"
                termination_reason = "Neural network failed (no moves)"
                break
            move_uci = np.random.choice(moves, p=probs)
            move = chess.Move.from_uci(move_uci)
            print(f"{player} plays: {move_uci}")
        else:
            player = "Black (Stockfish)"
            print(f"{player}'s turn to move...")
            best_move_raw = get_stockfish_best_move(board.fen(), depth=15)
            if best_move_raw:
                best_move = extract_best_move(best_move_raw)
                try:
                    move = chess.Move.from_uci(best_move)
                    if move in board.legal_moves:
                        print(f"{player} plays: {best_move}")
                    else:
                        print("Stockfish returned illegal move. Ending game.")
                        result = "1-0"
                        termination_reason = "Stockfish illegal move"
                        break
                except Exception as e:
                    print(f"Failed to apply Stockfish move: {e}")
                    result = "1-0"
                    termination_reason = "Stockfish move application failed"
                    break
            else:
                print("Stockfish failed to return a move.")
                result = "1-0"
                termination_reason = "Stockfish API failure"
                break
        
        board.push(move)
        move_count += 1
    
    # Assign result and termination reason if not already set
    if 'result' not in locals():
        result = board.result()
        if board.is_checkmate():
            termination_reason = "Checkmate"
        elif board.is_stalemate():
            termination_reason = "Stalemate"
        elif board.is_insufficient_material():
            termination_reason = "Insufficient material"
        elif board.is_seventyfive_moves():
            termination_reason = "75-move rule"
        elif board.is_fivefold_repetition():
            termination_reason = "Fivefold repetition"
        else:
            termination_reason = "Other draw condition"
    
    white_points, black_points = evaluate_material(board)
    duration = time.time() - start_time
    
    print(f"\nGame {game_number + 1} Over!")
    print(f"Final FEN: {board.fen()}")
    print(f"Final material points: White={white_points}, Black={black_points}")
    print(f"Result: {result}")
    print(f"Termination reason: {termination_reason}")
    
    return {
        "game_number": game_number + 1,
        "result": result,
        "termination_reason": termination_reason,
        "white_points": white_points,
        "black_points": black_points,
        "move_count": move_count,
        "duration_seconds": round(duration, 2)
    }

async def main():
    load_model()
    num_games = 5
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    game_metrics = []
    
    for game_number in range(num_games):
        metrics = await play_game(game_number)
        results[metrics["result"]] += 1
        game_metrics.append(metrics)
        print(f"\nProgress: {game_number + 1}/{num_games} games completed. Results: {results}")
    
    print("\nAll 5 games completed!")
    print(f"Final results - White (Neural Network) wins: {results['1-0']}, Black (Stockfish) wins: {results['0-1']}, Draws: {results['1/2-1/2']}")
    print("\nGame Metrics:")
    print("-" * 80)
    print(f"{'Game':<8} {'Result':<10} {'Termination Reason':<30} {'White Pts':<12} {'Black Pts':<12} {'Moves':<8} {'Duration (s)':<12}")
    print("-" * 80)
    for metrics in game_metrics:
        print(f"{metrics['game_number']:<8} {metrics['result']:<10} {metrics['termination_reason']:<30} {metrics['white_points']:<12} {metrics['black_points']:<12} {metrics['move_count']:<8} {metrics['duration_seconds']:<12}")
    print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main())