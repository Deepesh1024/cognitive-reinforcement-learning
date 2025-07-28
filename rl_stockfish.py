import chess
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List, Tuple, Dict
import copy
import math
import signal
import sys
import os
import pickle
from multiprocessing import Pool

# Neural Network Architecture (Simplified Policy + Value)
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(14, 128, kernel_size=3, padding=1)  # Reduced channels
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(64) for _ in range(5)])  # Reduced to 5 blocks
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4096),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(8 * 8, 128),  # Reduced size
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        for res_block in self.res_blocks:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

class ResBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x += residual
        return torch.relu(x)

# Board to Tensor Conversion
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = np.zeros((14, 8, 8), dtype=np.float32)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for i, piece_type in enumerate(piece_types):
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == piece_type:
                rank, file = divmod(square, 8)
                if piece.color == chess.WHITE:
                    planes[i][rank][file] = 1.0
                else:
                    planes[i + 6][rank][file] = 1.0
    planes[12] = board.castling_rights / 15.0
    planes[13] = 1.0 if board.turn == chess.WHITE else 0.0
    return torch.tensor(planes, dtype=torch.float32).unsqueeze(0)

# Move Encoding
def move_to_index(move: chess.Move, board: chess.Board) -> int:
    return move.from_square * 64 + move.to_square

def index_to_move(index: int, board: chess.Board) -> chess.Move:
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)

# Monte Carlo Tree Search
class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

class MCTS:
    def __init__(self, model: ChessNet, c_puct: float = 1.0, n_simulations: int = 100):
        self.model = model
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.eval_cache = {}  # Cache for board evaluations

    def search(self, board: chess.Board) -> Tuple[chess.Move, np.ndarray]:
        root = MCTSNode(board)
        board_key = board.fen()
        for _ in range(self.n_simulations):
            node = root
            search_board = copy.deepcopy(board)
            while node.children:
                node, move = self.select_child(node)
                search_board.push(move)
            if not search_board.is_game_over():
                fen = search_board.fen()
                if fen in self.eval_cache:
                    policy, value = self.eval_cache[fen]
                else:
                    try:
                        policy, value = self.model(board_to_tensor(search_board).to(device))
                        policy = policy.detach().cpu().numpy()[0]
                        value = value.detach().cpu().numpy()[0][0]
                        self.eval_cache[fen] = (policy, value)
                    except Exception as e:
                        print(f"MCTS simulation error: {e}")
                        value = 0.0
                        policy = np.zeros(4096)
                legal_moves = list(search_board.legal_moves)
                for move in legal_moves:
                    idx = move_to_index(move, search_board)
                    prior = policy[idx]
                    search_board.push(move)
                    node.children[move] = MCTSNode(copy.deepcopy(search_board), node, move)
                    node.children[move].prior = prior
                    search_board.pop()
            else:
                value = 0.0 if (search_board.is_stalemate() or
                                search_board.is_insufficient_material() or
                                search_board.is_fifty_moves()) else -1.0
            self.backpropagate(node, value)
        visits = np.array([child.visits for child in root.children.values()])
        moves = list(root.children.keys())
        probs = visits / visits.sum()
        return random.choices(moves, weights=probs)[0], probs

    def select_child(self, node: MCTSNode) -> Tuple[MCTSNode, chess.Move]:
        best_score = -float('inf')
        best_move = None
        best_child = None
        for move, child in node.children.items():
            score = child.value / (child.visits + 1e-8) + self.c_puct * child.prior * math.sqrt(node.visits) / (1 + child.visits)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_child, best_move

    def backpropagate(self, node: MCTSNode, value: float):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
            value = -value

# Chess Agent
class ChessAgent:
    def __init__(self, model_path: str = None):
        self.model = ChessNet().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.mcts = MCTS(self.model)
        self.memory = deque(maxlen=1000000)
        self.game_length = 200
        self.gamma = 0.99
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}. Starting with a fresh model.")
        else:
            print(f"No model file found at {model_path}. Starting with a fresh model.")

    def get_move(self, fen: str) -> str:
        try:
            board = chess.Board(fen)
            move, _ = self.mcts.search(board)
            return move.uci()
        except Exception as e:
            print(f"Error in get_move: {e}")
            return None

    def save_model(self, path: str):
        try:
            torch.save(self.model.state_dict(), path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def save_memory(self, path: str = "memory.pkl"):
        try:
            with open(path, "wb") as f:
                pickle.dump(list(self.memory), f)
            print(f"Memory saved to {path}")
        except Exception as e:
            print(f"Failed to save memory: {e}")

    def load_memory(self, path: str = "memory.pkl"):
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.memory.extend(pickle.load(f))
                print(f"Loaded memory from {path}")
        except Exception as e:
            print(f"Failed to load memory: {e}")

    def self_play(self, n_games: int) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        games_data = []
        for _ in range(n_games):
            board = chess.Board()
            game_states = []
            while not board.is_game_over() and board.ply() < self.game_length:
                try:
                    move, policy = self.mcts.search(board)
                    state = board_to_tensor(board)
                    board.push(move)
                    next_state = board_to_tensor(board) if not board.is_game_over() else None
                    game_states.append((state, policy, None, next_state))
                except Exception as e:
                    print(f"Self-play error: {e}")
                    break
            result = 0.0
            if board.is_checkmate():
                result = 1.0 if board.turn == chess.BLACK else -1.0
            elif (board.is_stalemate() or
                  board.is_insufficient_material() or
                  board.is_fifty_moves()):
                result = 0.0
            for i, (state, policy, _, next_state) in enumerate(game_states):
                self.memory.append((state, policy, result, next_state))
                result = -result
            games_data.extend(self.memory)
        return games_data

    def train(self, batch_size: int = 16, epochs: int = 2):
        self.model.train()
        for epoch in range(epochs):
            if len(self.memory) < batch_size:
                print(f"Epoch {epoch + 1}: Insufficient data in memory ({len(self.memory)}/{batch_size})")
                continue
            batch = random.sample(self.memory, batch_size)
            states, policies, values, next_states = zip(*batch)
            states = torch.cat(states).to(device)
            policies = torch.tensor(np.array(policies), dtype=torch.float32).to(device)
            values = torch.tensor(values, dtype=torch.float32).to(device).unsqueeze(1)
            self.optimizer.zero_grad()
            try:
                pred_policies, pred_values = self.model(states)
                policy_loss = -torch.mean(torch.sum(policies * torch.log(pred_policies + 1e-8), dim=1))
                value_loss = torch.mean((pred_values - values) ** 2)
                td_loss = 0.0
                for i, next_state in enumerate(next_states):
                    if next_state is not None:
                        with torch.no_grad():
                            _, next_value = self.model(next_state.to(device))
                        target = values[i] + self.gamma * next_value
                        td_loss += (pred_values[i] - target) ** 2
                td_loss = td_loss / batch_size if td_loss != 0.0 else torch.tensor(0.0).to(device)
                total_loss = policy_loss + value_loss + td_loss
                total_loss.backward()
                self.optimizer.step()
                print(f"Epoch {epoch + 1}: Policy Loss: {policy_loss.item():.4f}, "
                      f"Value Loss: {value_loss.item():.4f}, TD Loss: {td_loss.item():.4f}, "
                      f"Total Loss: {total_loss.item():.4f}")
            except Exception as e:
                print(f"Training error in epoch {epoch + 1}: {e}")

# Stockfish API Functions
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
    if raw_bestmove.startswith("bestmove"):
        return raw_bestmove.split()[1]
    return raw_bestmove.strip()

# Signal Handler for Ctrl+C
def signal_handler(sig, frame, agent: ChessAgent, model_path: str):
    print("\nCtrl+C detected. Saving model and memory...")
    agent.save_model(model_path)
    agent.save_memory("memory.pkl")
    sys.exit(0)

# Parallel Self-Play
def play_single_game(agent):
    return agent.self_play(n_games=1)

def parallel_self_play(agent, n_games):
    with Pool(processes=4) as pool:
        results = pool.starmap(play_single_game, [(agent,) for _ in range(n_games)])
    return [item for sublist in results for item in sublist]

# Main Game Loop
def main():
    model_path = "chess_model.pth"
    agent = ChessAgent(model_path=model_path)
    agent.load_memory("memory.pkl")  # Load existing memory if available
    game_count = 0
    has_won = False

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, agent, model_path))

    print(f"Using device: {device}")
    while not has_won:
        game_count += 1
        print(f"\nStarting Game {game_count}")
        board = chess.Board()
        print("Starting FEN:", board.fen())

        while not board.is_game_over():
            print("\nCurrent position:")
            print(board)
            print("FEN:", board.fen())

            if board.turn == chess.WHITE:
                print("RL Agent is thinking...")
                move_uci = agent.get_move(board.fen())
                if move_uci:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            print(f"RL Agent plays: {move_uci}")
                            board.push(move)
                        else:
                            print("Illegal move by RL Agent. Game aborted.")
                            break
                    except Exception as e:
                        print(f"Invalid move by RL Agent: {e}")
                        break
                else:
                    print("RL Agent failed to return a move. Game aborted.")
                    break
            else:
                print("Stockfish is thinking...")
                best_move_raw = get_stockfish_best_move(board.fen(), depth=15)
                if best_move_raw:
                    best_move = extract_best_move(best_move_raw)
                    try:
                        move = chess.Move.from_uci(best_move)
                        print(f"Stockfish plays: {best_move}")
                        board.push(move)
                    except Exception as e:
                        print(f"Failed to apply Stockfish move: {e}")
                        break
                else:
                    print("Stockfish failed to return a move.")
                    break

        print("\nGame Over!")
        print("Final FEN:", board.fen())
        result = board.result()
        print("Result:", result)
        if result == "1-0":
            has_won = True
            print("RL Agent has beaten Stockfish!")
            agent.save_model(model_path)
            agent.save_memory("memory.pkl")
        else:
            print("Training agent to improve...")
            try:
                games_data = parallel_self_play(agent, n_games=2)
                agent.train(batch_size=16, epochs=2)
                agent.save_model(model_path)
                agent.save_memory("memory.pkl")
            except Exception as e:
                print(f"Training failed: {e}")

# Device Setup and Execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    main()