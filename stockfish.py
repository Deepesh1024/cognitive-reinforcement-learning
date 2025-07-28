import requests
import chess

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

def main():
    board = chess.Board()  
    print("Starting FEN:", board.fen())

    while not board.is_game_over():
        print("\nCurrent position:")
        print(board)
        print("FEN:", board.fen())

        if board.turn == chess.WHITE:
            user_move = input("Your move (UCI format, e.g. e2e4): ").strip()
            try:
                move = chess.Move.from_uci(user_move)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move. Try again.")
            except Exception as e:
                print("Invalid input:", e)
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
                    print("Failed to apply move:", e)
                    break
            else:
                print("Stockfish failed to return a move.")
                break

    print("\nGame Over!")
    print("Final FEN:", board.fen())
    print("Result:", board.result())

if __name__ == "__main__":
    main()
