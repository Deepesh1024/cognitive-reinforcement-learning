import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import signal
import sys
import asyncio

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

# Training and Self-Play
MODEL_FILE = "chess_model.pth"
model = ChessNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Initializing new model.")
            model.__init__()

def save_model():
    try:
        torch.save(model.state_dict(), MODEL_FILE)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

def signal_handler(sig, frame):
    print("\nCtrl+C detected. Saving model...")
    save_model()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

async def play_game(game_number):
    model.train()
    board = chess.Board()
    mcts = MCTS(model, num_simulations=100)
    game_data = []
    move_count = 0
    
    print(f"\nGame {game_number + 1} - Starting FEN: {board.fen()}")
    
    while not board.is_game_over():
        fen_parts = board.fen().split()
        fullmove_number = int(fen_parts[5])
        if fullmove_number >= 50:
            white_points, black_points = evaluate_material(board)
            if white_points > black_points:
                result = "1-0"
                value = 1.0
            elif black_points > white_points:
                result = "0-1"
                value = -1.0
            else:
                result = "1/2-1/2"
                value = 0.0
            print(f"Game {game_number + 1} ended due to 100-move rule. Material: White={white_points}, Black={black_points}")
            break
        
        print(f"\nGame {game_number + 1} - Current position:")
        print(board)
        print(f"FEN: {board.fen()}")
        
        player = "White" if board.turn == chess.WHITE else "Black"
        print(f"{player}'s turn to move...")
        
        move_probs, root = mcts.search(board)
        moves = list(move_probs.keys())
        probs = list(move_probs.values())
        if not moves:
            print("No valid moves found.")
            break
        
        move_uci = np.random.choice(moves, p=probs)
        move = chess.Move.from_uci(move_uci)
        
        move_idx = mcts.move_to_index(move)
        state = mcts.board_to_tensor(board).numpy()
        game_data.append((state, move_idx, None))
        
        print(f"{player} plays: {move_uci}")
        board.push(move)
        move_count += 1
    
    # Assign rewards
    if not board.is_game_over() and 'result' not in locals():
        result = board.result()
        if result == "1-0":
            value = 1.0
        elif result == "0-1":
            value = -1.0
        else:
            value = 0.0
    
    # Update game data with values
    for i, (state, move_idx, _) in enumerate(game_data):
        game_data[i] = (state, move_idx, value if (board.turn == chess.WHITE) == (i % 2 == 0) else -value)
    
    # Train the model
    for state, move_idx, value_target in game_data:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        move_idx_tensor = torch.tensor([move_idx], dtype=torch.long)
        value_target_tensor = torch.tensor(value_target, dtype=torch.float32)
        
        optimizer.zero_grad()
        policy_logits, value = model(state_tensor)
        policy_loss = F.cross_entropy(policy_logits, move_idx_tensor)
        value_loss = F.mse_loss(value.squeeze(), value_target_tensor)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()
    
    print(f"\nGame {game_number + 1} Over!")
    print(f"Final FEN: {board.fen()}")
    print(f"Result: {result}")
    save_model()
    return result

async def main():
    load_model()
    num_games = 100
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    
    for game_number in range(num_games):
        result = await play_game(game_number)
        results[result] += 1
        print(f"\nProgress: {game_number + 1}/{num_games} games completed. Results: {results}")
    
    print("\nAll 100 games completed!")
    print(f"Final results - White wins: {results['1-0']}, Black wins: {results['0-1']}, Draws: {results['1/2-1/2']}")
    save_model()

if __name__ == "__main__":
    asyncio.run(main())