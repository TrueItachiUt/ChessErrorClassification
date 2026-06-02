import numpy as np
import time
from typing import Optional, Literal
from IPython.display import display, clear_output
from minimal_lc0_for_research.leela_board import LeelaBoard
from chess import Board
from backend_tools import *
from Dataset import threshold, targets

russian_targets = [
    'Открытый король', 'Жертва', 'Висящая фигура', 'Вилка', 
    'Уничтожение защиты', 'Связка', 'Тихий ход', 'Промежуточный ход', 'Отвлечение'
]

context_labels = {
    "Blunder_opp_used": "Игрок ошибся, его соперник этим воспользовался",
    "Blunder_opp_not_used": "Игрок ошибся, но его соперник этим не воспользовался",
    "Missed_blunder": "Игрок не воспользовался возможностью значительно улучшить свою позицию"
}

def show_prediction(fen: str, moves: list[str], target_bin: bool = None, target_class: int = None, demonstrate: bool = False):
    delay = 2
    positions, evals = prepare_for_model([fen], [moves])
    preds = dnn_model((positions, evals))

    bin_prob = float(preds['binary'][0][1].numpy())
    is_strike = bin_prob > threshold

    pred_class_idx = None
    pred_class_name = None
    if is_strike:
        pred_class_idx = int(np.argmax(preds['multiclass'][0].numpy()))
        if 0 <= pred_class_idx < len(targets):
            pred_class_name = targets[pred_class_idx]

    board = LeelaBoard(fen=fen)
    display(board)
    time.sleep(delay)

    if demonstrate:
        for move in moves:
            board.push_uci(move)
            display(board)
            clear_output(wait=True)
            time.sleep(delay)
        display(board)

    print(f"Moves: {' '.join(moves)}")
    print(f"Binary Strike Prob: {bin_prob:.3f}")
    if is_strike:
        print(f"Predicted Class: {pred_class_name} (Idx: {pred_class_idx})")
    else:
        print("Prediction: No tactical strike detected.")
        
    if target_bin is not None:
        print("Blunder" if target_bin else "Nothing")
        if target_class is not None and 0 <= target_class < len(targets):
            print(f"Target Class: {targets[target_class]}")

def replay_report(game_data: list[dict], start_fen: str = None,
                clean_delay: float = 1.0, error_delay: float = 5.0):
    """
    Replays the game move by move, showing evaluations and analyzing mistakes.
    """
    if start_fen is None:
        start_fen = Board().fen()
        
    board = Board(fen=start_fen)
    display(LeelaBoard(fen=board.fen()))
    
    for move_idx, data in enumerate(game_data):
        clear_output(wait=True)
        board.push_uci(data['uci'])
        display(LeelaBoard(fen=board.fen()))
        
        eval_str = format_eval(data['eval_after'])
        print(f"Ход {move_idx//2}: {data['side']} | {data['uci']} | Оценка: {eval_str}")
        
        if data['is_error']:
            eval_before_str = format_eval(data['eval_before'])
            print(f"🚨 ОШИБКА! Оценка упала с {eval_before_str} до {eval_str}")
            
            if data['correct_move']:
                print(f"✅ Правильный ход был: {data['correct_move']}")
            if data['scenario']:
                print(f"📝 Контекст: {context_labels.get(data['scenario'], data['scenario'])}")
            if data['tactic_class'] is not None and 0 <= data['tactic_class'] < len(targets):
                print(f"⚔️ Тактика: {russian_targets[data['tactic_class']]}")
            
            time.sleep(error_delay)
        else:
            time.sleep(clean_delay)
            
    print("✅ Анализ партии завершен.")

def interaction(
    fen: str = None,
    moves: list[str] = None,
    evals: list[float] = None,
    demonstrate: bool = False,
    default_fen: str = "8/1p1r4/p2k1ppp/N1bn1p2/2B2P2/P4PPP/1P2R1K1/8 w - - 9 31",
    default_moves: list[str] = ["a5b3", "d5e3", "e2e3", "c5e3"]
) -> list[dict] | None:
    
    if fen is None:
        try: inp = input("FEN (press Enter for default): ").strip()
        except EOFError: inp = ""
        fen = inp if inp else default_fen

    if moves is None:
        try: inp = input("Moves (space-separated SAN/UCI, press Enter for default): ").strip()
        except EOFError: inp = ""
        moves = inp.split() if inp else default_moves

    try:
        uci_moves = to_uci_moves([fen], [moves])[0]
    except Exception as e:
        print(f"❌ Move conversion failed: {e}")
        return None

    try:
        check_data(fen, uci_moves)
        print("✅ Input validated.")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")
        return None

    try:
        game_data = process_clean_game(uci_moves, fen)
    except Exception as e:
        print(f"❌ Decision pipeline failed: {e}")
        raise

    if demonstrate:
        replay_game(game_data, start_fen=fen)
    else:
        # Console summary if not demonstrating
        print("\n--- Краткий отчет ---")
        for data in game_data:
            icon = "🚨" if data['is_error'] else "✅"
            eval_str = format_eval(data['eval_after'])
            line = f"[{data['move_idx']}] {icon} {data['side']} {data['uci']} | Оценка: {eval_str}"
            
            if data['is_error']:
                if data['correct_move']:
                    line += f" | Лучший ход: {data['correct_move']}"
                if data['scenario']:
                    line += f" | {context_labels.get(data['scenario'], '')}"
            print(line)
            
    return game_data

if __name__ == "__main__":
    # Example usage
    game_uci = ['d2d4', 'g8f6', 'g1f3', 'c7c5', 'c2c3', 'd8c7', 'c1g5', 'f6e4', 'g5h4', 'g7g6']
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    game_data = process_clean_game(game_uci, test_fen)
    print(game_data)
    #replay_report(game_data, start_fen=test_fen, clean_delay=0.5, error_delay=2.0)
    print("\n\n\nFor replay use interactive notebooks")