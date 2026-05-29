import numpy as np
import time
from IPython.display import display,clear_output
from minimal_lc0_for_research.leela_board import LeelaBoard
from backend_tools import prepare_for_model, dnn_model, to_uci_moves, check_data, decision
from Dataset import threshold, targets
import tensorflow as tf

def show_prediction(fen: str, moves: list[str], target_bin: bool = None, target_class: int = None, demonstrate: bool = False):
    delay = 0.8
    
    # 1. Prepare & Infer
    positions, evals = prepare_for_model([fen], [moves])
    print(f"evals {evals}\n\n")
    preds = dnn_model((positions, evals))
    
    # 2. Extract Binary Probability
    bin_prob = float(preds['binary'][0][1].numpy())
    is_strike = bin_prob > threshold
    # 3. Extract Multiclass ONLY if strike is predicted
    pred_class_idx = None
    pred_class_name = None
    if is_strike:
        pred_class_idx = int(np.argmax(preds['multiclass'][0].numpy()))
        if 0 <= pred_class_idx < len(targets):
            pred_class_name = targets[pred_class_idx]

    # 4. Visualization
    board = LeelaBoard(fen=fen)
    display(board)
    time.sleep(delay)
    
    if demonstrate:
        for move in moves:
            board.push_uci(move)
            display(board)
            clear_output(wait=True)
            time.sleep(delay)

    # 5. Console Output
    print(f"Moves: {' '.join(moves)}")
    print(f"Binary Strike Prob: {bin_prob:.3f}")
    
    if is_strike:
        print(f"Predicted Class: {pred_class_name} (Idx: {pred_class_idx})")
    else:
        print("Prediction: No tactical strike detected.")
        
    if target_bin is not None:
        print(f"Target: Strike" if target_bin else "Target: Clean")
        if target_class is not None and 0 <= target_class < len(targets):
            print(f"Target Class: {targets[target_class]}")


def human_readable_report(decision_results: list) -> list[dict]:
    """Converts raw decision() output to a structured, frontend-ready format."""
    scenario_labels = {
        "Blunder_opp_used": "Opponent capitalized on your mistake",
        "Blunder_opp_not_used": "Mistake made, opponent missed it",
        "Missed_blunder": "You missed opponent's mistake"
    }
    
    report = []
    for idx, (is_strike, cls_idx, scenario) in enumerate(decision_results):
        tactic_class = None
        if is_strike and cls_idx is not None and 0 <= cls_idx < len(targets):
            tactic_class = targets[cls_idx]
            
        report.append({
            "move_index": idx,
            "is_strike": bool(is_strike),
            "tactic_class": tactic_class,
            "context_key": scenario if scenario else "clean",
            "ui_message": scenario_labels.get(scenario, "No tactical blunder detected.")
        })
    return report


def interaction(
    fen: str = None,
    moves: list[str] = None,
    evals: list[float] = None,
    demonstrate: bool = False,
    default_fen: str = "8/1p1r4/p2k1ppp/N1bn1p2/2B2P2/P4PPP/1P2R1K1/8 w - - 9 31",
    default_moves: list[str] = ["a5b3", "d5e3", "e2e3", "c5e3"]
) -> list[dict] | None:

    # 1. Gather inputs
    if fen is None:
        try: inp = input("FEN (press Enter for default): ").strip()
        except EOFError: inp = ""
        fen = inp if inp else default_fen

    if moves is None:
        try: inp = input("Moves (space-separated SAN/UCI, press Enter for default): ").strip()
        except EOFError: inp = ""
        moves = inp.split() if inp else default_moves

    # 2. Convert & Validate
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

    # 3. Decision Pipeline
    try:
        dec_results = decision([fen], [uci_moves], [evals] if evals is not None else None)
    except Exception as e:
        print(f"❌ Decision pipeline failed: {e}")
        raise

    # 4. Format & Output
    report = human_readable_report(dec_results)

    if demonstrate:
        show_prediction(fen, uci_moves, demonstrate=True)

    for r in report:
        icon = "🚨" if r['is_strike'] else "✅"
        cls_str = f" | Class: {r['tactic_class']}" if r['tactic_class'] else ""
        print(f"[{r['move_index']}] {icon} {r['ui_message']}{cls_str} | Key: {r['context_key']}")

    return report


if __name__ == "__main__":
    test_fen = "rn2k1nr/pp2ppb1/6pp/q2N4/3P3B/3QP3/PP3PPP/R3K1NR w KQkq - 1 11"
    test_moves = "d3c3 a5d5 c3c8 d5d8 c8b7 b8d7".split()
    test_target = 2
    
    show_prediction(test_fen, test_moves, target_bin=True, target_class=test_target, demonstrate=False)
    #interaction(test_fen, test_moves)
    print(f"Correct class is {targets[test_target]}")