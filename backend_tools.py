from maia2 import model, dataset, inference
from chess import Board
from Model import *
from typing import Union, Optional
from Dataset import threshold, error_delta
from minimal_lc0_for_research.leela_board import LeelaBoard
from config import PATH_TO_STOCKFISH
import numpy as np
from stockfish import Stockfish

tf.config.experimental.enable_op_determinism()

_eng_cache = None
def _get_engine():
    global _eng_cache
    if _eng_cache is None:
        _eng_cache = Stockfish(path=PATH_TO_STOCKFISH, depth=1)
    return _eng_cache

maia2_model = model.from_pretrained(type="rapid", device="cpu")
dnn_model = CNNLSTM()
prepared = inference.prepare()

def generate_moves(fen, n_moves=4):
    board = Board(fen=fen)
    moves = []
    for _ in range(n_moves):
        if board.is_game_over():
            break
            
        legal = list(board.legal_moves)
        if not legal:
            break
            
        try:
            move_probs, _ = inference.inference_each(maia2_model, prepared, board.fen(), 1900, 1900, temperature=0)
            move = max(move_probs, key=move_probs.get)
        except Exception:
            move = legal[0].uci()  # Deterministic fallback on inference failure
            
        moves.append(move)
        board.push_uci(move)
        
    return moves



def prepare_for_model(fens, moves, evs=None):
    batch_size = len(fens)
    a = np.zeros((batch_size, 5, 8, 8, 112))
    f = evs is None
    if f:
        evs = np.zeros((batch_size, 5))
    else:
        tt = []
        for i in range(batch_size):
            k = np.array(evs[i])
            if len(k)<5:
                k = np.pad(k, (0, 5-len(k)))
                print(k)
            tt.append(k)
        evs = np.array(tt)
    
    for b, fen in enumerate(fens):
        board = LeelaBoard(fen=fen)
        if f: _get_engine().set_fen_position(fen)
        for i in range(min(4, len(moves[b]))):
            if i != 0: 
                board.push_uci(moves[b][i-1])
                if f:
                    _get_engine().make_moves_from_current_position([moves[b][i-1]])
            a[b, i] = np.moveaxis(board.lcz_features(), 0, -1)
            if f:
                evs[b, i] = _get_engine().get_evaluation()['value']/100
            
    return a, evs

def _get_fen(pr_fen, move):
    b = Board(pr_fen); b.push_uci(move)
    return b.fen()

def decision(fens: list[str], moves: list, evals=None
) -> list[tuple[bool, Optional[int], Optional[Union["Blunder_opp_used", "Blunder_opp_not_used", "Missed_blunder"]]]]:
    '''
    Batched main decision function with masking to avoid re-processing classified instances.

    Input:
        fens: list of FENs BEFORE the critical moves
        moves: list of sequences (n<=4 +1 moves)
        evals: list of position evaluations (list of lists), optional

    Output:
        Batch size list of elements
            tuple of 
                Strike flag (bool)
                Class of strike (int)
                One of three elems: 
                "Blunder_opp_used" (player made a mistake, opponent used it), 
                "Blunder_opp_not_used" (player made a mistake, opponent didnt use it),    
                "Missed_blunder" (player didnt use opponents mistake)
    '''

    batch_size = len(fens)

    results = [(False, None, None) for _ in range(batch_size)]
    active_mask = np.ones(batch_size, dtype=bool)

    # 0. Baseline filtering
    if evals is not None:
        # Create mask for rows that have at least 3 elements and meet the threshold condition
        naive_mask = np.array([
            (evals[i][0] - evals[i][2]) < threshold if len(evals[i]) >= 3 else False 
            for i in range(batch_size)
        ], dtype=bool)

        active_indices = np.where(naive_mask)[0]

        if len(active_indices) > 0:
            fens_0 = [fens[i] for i in active_indices]
            moves_0 = [moves[i] for i in active_indices]
            # Filter evals list using list comprehension
            evals_0 = [evals[i] for i in active_indices]

            pos_0, evs_0 = prepare_for_model(fens_0, moves_0, evals_0)

            active_mask[naive_mask] = False

            preds0 = dnn_model((pos_0, evs_0))['multiclass'].numpy().argmax(axis=1)
            for j, ind in enumerate(active_indices):
                results[ind] = (True, preds0[j], "Blunder_opp_not_used")

    # 1. Main game continuation (k -> k+5)
    active_indices = np.where(active_mask)[0]
    if len(active_indices) == 0:
        return results

    k_fens = [_get_fen(fens[i], moves[i][0]) for i in active_indices]

    # Filter evals: take rows active_indices, then columns 1: (skip first element)
    if evals is not None:
        evals_filtered = [evals[i][1:] for i in active_indices]
    else:
        evals_filtered = None

    pos1, evs1 = prepare_for_model(
        k_fens,
        [moves[i][1:] for i in active_indices],
        evals_filtered
    )

    preds1 = dnn_model((pos1, evs1))

    bin1 = preds1['binary'].numpy()[:, 1] >= threshold
    mult1 = preds1['multiclass'].numpy().argmax(axis=1)

    hit_mask = bin1

    for j, i in enumerate(active_indices[hit_mask]):
        results[i] = (
            True,
            int(mult1[hit_mask][j]),
            "Blunder_opp_used"
        )

    active_mask[active_indices[hit_mask]] = False

    # 2. Previous Maia continuation (k-1)
    active_indices = np.where(active_mask)[0]

    if len(active_indices):
        prv_fens = [fens[i] for i in active_indices]
        prv_moves = [generate_moves(f) for f in prv_fens]

        pos2, evs2 = prepare_for_model(prv_fens, prv_moves, None)

        preds2 = dnn_model((pos2, evs2))

        bin2 = preds2['binary'].numpy()[:, 1] >= threshold
        mult2 = preds2['multiclass'].numpy().argmax(axis=1)

        hit_mask = bin2

        for j, i in enumerate(active_indices[hit_mask]):
            results[i] = (
                True,
                int(mult2[hit_mask][j]),
                "Missed_blunder"
            )

        active_mask[active_indices[hit_mask]] = False

    # 3. Next Maia continuation (k+1)
    active_indices = np.where(active_mask)[0]

    if len(active_indices):
        nxt_fens = [
            _get_fen(_get_fen(fens[i], moves[i][0]), moves[i][1])
            for i in active_indices
        ]

        nxt_moves = [generate_moves(f) for f in nxt_fens]

        pos3, evs3 = prepare_for_model(nxt_fens, nxt_moves, None)

        preds3 = dnn_model((pos3, evs3))

        bin3 = preds3['binary'].numpy()[:, 1] >= threshold
        mult3 = preds3['multiclass'].numpy().argmax(axis=1)

        hit_mask = bin3

        for j, i in enumerate(active_indices[hit_mask]):
            results[i] = (
                True,
                int(mult3[hit_mask][j]),
                "Blunder_opp_not_used"
            )

        active_mask[active_indices[hit_mask]] = False

    return results

def process_clean_game(moves: list[str], fen: str = None) -> dict[int, tuple[int, Optional[Union["Blunder_opp_used", "Blunder_opp_not_used", "Missed_blunder"]]]]:
    if fen is None:
        fen = Board().fen()
    _get_engine().set_fen_position(fen)
    n = len(moves)

    # fens[i] & evals[i] represent state BEFORE move i
    fens = [fen] + [None] * n
    evals = np.empty(n + 1)
    evals[0] = _get_engine().get_evaluation()['value'] / 100

    for i, m in enumerate(moves):
        _get_engine().make_moves_from_current_position([m])
        evals[i+1] = _get_engine().get_evaluation()['value'] / 100
        fens[i+1] = _get_engine().get_fen_position()

    q_f, q_m, q_evs, keys = [], [], [], []
    for i in range(3, n-2):
        if abs(evals[i+1] - evals[i]) >= error_delta:
            q_f.append(fens[i])          # FEN before the blundering move
            q_m.append(moves[i : i+4])   # Blunder + next 3 moves
            q_evs.append(evals[i : i+4]) # Corresponding evals
            keys.append(i + 1)           # 1-based move index

    if not q_f:
        return {}

    return dict(zip(keys, decision(q_f, q_m, q_evs)))

def check_data(fen: str, moves: list[str]) -> None:
    """Validates FEN and sequential UCI moves. Raises ValueError on failure."""
    try:
        board = Board(fen=fen)
    except ValueError as e:
        raise ValueError(f"Invalid FEN: {e}") from e

    if not board.is_valid():
        raise ValueError("FEN encodes an illegal position.")

    for i, uci in enumerate(moves):
        try:
            move = board.parse_uci(uci)
        except (ValueError, IndexError):
            raise ValueError(f"Move {i} '{uci}' is malformed.")

        if not board.is_legal(move):
            raise ValueError(f"Move {i} '{uci}' is illegal.")

        board.push(move)

def to_uci_moves(fens: list[str], moves: list[list[str]]) -> list[list[str]]:
    """Converts batched SAN/PGN or UCI move sequences to UCI."""
    uci_out = []
    for fen, seq in zip(fens, moves):
        board = Board(fen)
        uci_seq = []
        for m in seq:
            try:
                mv = board.parse_uci(m)
            except ValueError:
                mv = board.parse_san(m)  # Fallback for PGN/SAN (e.g., "Nf3", "O-O")
            except TypeError:  # Move class instance
                mv = m
            uci_seq.append(mv.uci())
            board.push(mv)
        uci_out.append(uci_seq)
    return uci_out

if __name__=='__main__':
    test_fen = 'rnbqkbnr/pppp1ppp/8/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq d3 0 2'
    
    # 1. Test SAN/PGN -> UCI conversion
    san_seq = [['e4', 'Ne5', 'Nc6', 'd5']]
    uci_seq = to_uci_moves([test_fen], san_seq)
    print("1. SAN -> UCI:", uci_seq[0])
    
    # 2. Test FEN & UCI validation
    try:
        check_data(test_fen, uci_seq[0])
        print("2. check_data: Valid ✅")
    except ValueError as e:
        print(f"2. check_data: ❌ {e}")
        
    # 3. Test decision() with & without evals
    full_moves = generate_moves(test_fen, n_moves=5)
    q_f, q_m = [test_fen], [full_moves[:4]]
    
    res_no_evs = decision(q_f, q_m, evals=None)
    print("3. decision (evals=None):", res_no_evs)
    
    # Mock evals to trigger naive_mask branch in decision
    q_evs = np.array([[0.5, -3.0, 0.2, -0.1]]) 
    res_with_evs = decision(q_f, q_m, evals=q_evs)
    print("4. decision (with evals):", res_with_evs)
    
    # 5. End-to-end pipeline
    report = process_clean_game(full_moves, test_fen)
    print("5. process_clean_game:", report)
    
    print("\n✅ Backend validation complete.")