import numpy as np
import time
from typing import Optional, Literal
from IPython.display import display, clear_output
from minimal_lc0_for_research.leela_board import LeelaBoard
from chess import Board
import backend_tools as bt
from Dataset import threshold, targets, russian_targets, context_labels

def show_prediction(fen: str, moves: list[str], target_bin: bool = None, target_class: int = None, demonstrate: bool = False):
    if bt.dnn_model is None:
        bt.load_models()
    delay = 2
    positions, evals = bt.prepare_for_model([fen], [moves])
    preds = bt.dnn_model((positions, evals))

    bin_prob = float(preds['binary'][0][1].numpy())
    is_strike = bin_prob  > threshold

    pred_class_idx = None
    pred_class_name = None
    if is_strike:
        pred_class_idx = int(np.argmax(preds['multiclass'][0].numpy()))
        if 0  <= pred_class_idx  < len(targets):
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
        print("Blunder " if target_bin else "Nothing ")
        if target_class is not None and 0  <= target_class  < len(targets):
            print(f"Target Class: {targets[target_class]}")

def print_game_report(game_data: list[dict], show_only_errors: bool = True) -> None:
    """
    Выводит анализ партии в удобном для человека виде.
    По умолчанию показывает только ходы с ошибками.
    """
    global context_labels, russian_targets

    def _fmt_eval(d: dict) -> str:
        if not d: return "N/A"
        if d.get('type') == 'mate':
            return f"+M{d['value']}" if d['value'] > 0 else f"-M{abs(d['value'])}"
        return f"{d.get('value', 0) / 100:+.2f}"

    print("\n" + "=" * 50)
    print("📊 ОТЧЕТ АНАЛИЗА ПАРТИИ")
    print("=" * 50 + "\n")

    for i, move_data in enumerate(game_data):
        if show_only_errors and not move_data.get("is_error"):
            continue

        move_num = i + 1
        side = move_data["side"]
        uci = move_data["uci"]
        
        eval_before_str = _fmt_eval(move_data["eval_before"])
        eval_after_str = _fmt_eval(move_data["eval_after"])

        if move_data["is_error"]:
            print(f"🚨 Ход {move_num} ({side}): {uci}")
            print(f"   📉 Оценка: {eval_before_str} ➔ {eval_after_str}")
            
            scenario = move_data.get("scenario")
            if scenario in context_labels:
                print(f"   📝 Контекст: {context_labels[scenario]}")
            
            tactic_class = move_data.get("tactic_class")
            if tactic_class is not None and 0 <= tactic_class < len(russian_targets):
                print(f"   ⚔️ Тактика: {russian_targets[tactic_class]}")
            
            correct_move = move_data.get("correct_move")
            if correct_move:
                print(f"   ✅ Лучший ход был: {correct_move}")
            print("-" * 50)
        else:
            # Краткий вывод для обычных ходов, если show_only_errors=False
            print(f"✅ Ход {move_num} ({side}): {uci} | Оценка: {eval_after_str}")

    print("=" * 50 + "\n")

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
        
        eval_str = bt.format_eval(data['eval_after'])
        print(f"Ход {move_idx//2}: {data['side']} | {data['uci']} | Оценка: {eval_str} ")
        
        if data['is_error']:
            eval_before_str = bt.format_eval(data['eval_before'])
            print(f"🚨 ОШИБКА! Оценка упала с {eval_before_str} до {eval_str} ")
            
            if data['correct_move']:
                print(f"✅ Правильный ход был: {data['correct_move']} ")
            if data['scenario']:
                print(f"📝 Контекст: {context_labels.get(data['scenario'], data['scenario'])} ")
            # Исправлено: проверка длины russian_targets вместо targets
            if data['tactic_class'] is not None and 0  <= data['tactic_class']  < len(russian_targets):
                print(f"⚔️ Тактика: {russian_targets[data['tactic_class']]} ")
            
            time.sleep(error_delay)
        else:
            time.sleep(clean_delay)
            
    print("✅ Анализ партии завершен. ")

def interaction(
fen: str = None,
moves: list[str] = None,
evals: list[float] = None,
demonstrate: bool = False,
default_fen: str = "8/1p1r4/p2k1ppp/N1bn1p2/2B2P2/P4PPP/1P2R1K1/8 w - - 9 31",
default_moves: list[str] = ["a5b3", "d5e3", "e2e3", "c5e3"]
) -> list[dict] | None:
    if fen is None:
        try: inp = input("FEN (press Enter for default):  ").strip()
        except EOFError: inp = " "
        fen = inp if inp else default_fen

    if moves is None:
        try: inp = input("Moves (space-separated SAN/UCI, press Enter for default):  ").strip()
        except EOFError: inp = " "
        moves = inp.split() if inp else default_moves

    try:
        uci_moves = bt.to_uci_moves([fen], [moves])[0]
    except Exception as e:
        print(f"❌ Move conversion failed: {e}")
        return None

    try:
        bt.check_data(fen, uci_moves)
        print("✅ Input validated.")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")
        return None

    try:
        game_data = bt.process_clean_game(uci_moves, fen)
    except Exception as e:
        print(f"❌ Decision pipeline failed: {e}")
        raise

    if demonstrate:
        # Исправлено: replay_game -> replay_report
        replay_report(game_data, start_fen=fen)
    else:
        # Console summary if not demonstrating
        print("\n--- Краткий отчет ---")
        # Исправлено: использование enumerate вместо отсутствующего data['move_idx']
        for i, data in enumerate(game_data):
            icon = "🚨 " if data['is_error'] else "✅ "
            eval_str = bt.format_eval(data['eval_after'])
            line = f"[{i+1}] {icon} {data['side']} {data['uci']} | Оценка: {eval_str} "
            
            if data['is_error']:
                if data['correct_move']:
                    line += f" | Лучший ход: {data['correct_move']} "
                if data['scenario']:
                    line += f" | {context_labels.get(data['scenario'], '')} "
            print(line)
            
    return game_data

if __name__ == "__main__":
    # Example usage
    game_uci = ['d2d4', 'g8f6', 'g1f3', 'c7c5', 'c2c3', 'd8c7', 'c1g5', 'f6e4', 'g5h4', 'g7g6']
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    game_data = bt.process_clean_game(game_uci, test_fen)
    print_game_report(game_data)
    #replay_report(game_data, start_fen=test_fen, clean_delay=0.5, error_delay=2.0)
    print("\n\n\nFor replay use interactive notebooks")