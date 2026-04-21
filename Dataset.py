'''This file has all information about current dataset
    Not to be confused with Dataset.ipynb, where all experiments and generation is run'''
from chess.pgn import read_game
import numpy as np
from minimal_lc0_for_research.leela_board import LeelaBoard
from tqdm import tqdm
from tensorflow.data import Dataset
import tensorflow as tf
from config import *
try:
    from stockfish import Stockfish
except ImportError:
    pass

target_lichess_classes = [
    'exposedKing',
    'sacrifice',
    'hangingPiece',
    'fork',
    'captureTheDefender',
    'pin',
    'quietMove',
    'intermezzo',
    'deflection'
]
additional_target_classes = [
    'planlessGame'
]
num_classes = len(target_lichess_classes)+len(additional_target_classes)+1
class_weight = 0.1 # proba of sample contain tactical strike
n_game_samples=2
n_moves = 5

def get_game_fens(batch_size=10):
    if not IS_PROJECT:
        raise ValueError("Trying to run data generation not from the computer")
    with open('data/lichess_elite_2022-01.pgn') as f:
        FENs = []
        all_moves = []
        # Target number of samples: batch_size adjusted by class_weight and samples per game
        target_samples = int(batch_size * (1 - class_weight) // n_game_samples)
        
        while len(FENs) < target_samples:
            game = read_game(f)
            moves = list(game.mainline_moves())
            total = len(moves)
            
            # Cannot sample if game is too short
            if total < n_moves:
                continue
                
            # Available indices: must have at least n_moves remaining after idx
            available = [i for i in range(total - n_moves + 1)]
            p = np.array([True] * len(available)).astype(bool)  # availability mask
            
            for _ in range(n_game_samples):
                # Check if any indices remain available
                if not np.any(p):
                    break
                    
                # Sample from available indices only
                idx_in_available = np.random.choice(np.where(p)[0])
                idx = available[idx_in_available]
                
                # Mark neighborhood as unavailable to prevent intersections
                # Block [idx - n_moves, idx + n_moves) in the original moves list
                for k, orig_idx in enumerate(available):
                    if abs(orig_idx - idx) < n_moves:
                        p[k] = False
                
                # Build position at idx
                board = game.board()
                for i in range(idx):
                    board.push(moves[i])
                FENs.append(board.fen())
                all_moves.append(moves[idx:idx + n_moves])  # Now guaranteed length == n_moves
                
    return FENs, all_moves

def get_binary_chunk(chunksize:int=10, class_weight:float=0.1, silent:bool=True, save:bool=False, filename='test.npz'):
    
    '''
    Creates chunk of data for binary classifier
    kwargs: 
    chunksize: number of instances of both classes
    class_weight: proportion between class 1 (positive) and class 2 (negative)
    '''

    if not IS_PROJECT:
        raise ValueError("Trying to run data generation not from the computer")
        
    path_to_binary = 'stockfish/stockfish-ubuntu-x86-64-avx2'
    engine = Stockfish(path=path_to_binary, depth=1)
    FENs, moves = get_game_fens(chunksize)
    if not silent: print(f"Len of FENs is {len(FENs)} len of moves is {len(moves)}")
    n_negative = int(chunksize*(1-class_weight))
    n_positive = int(chunksize*class_weight)
    negative_evals = np.zeros(shape=(n_negative, 5))
    negative_positions = np.zeros(shape=(n_negative, 5, 8, 8, 112))
    for i in range(len(FENs)):

        engine.set_fen_position(FENs[i])
        board = LeelaBoard(fen=FENs[i])

        for j in range(n_moves):
            cur_move = moves[i][j].uci()
            board.push_uci(cur_move)
            engine.make_moves_from_current_position([cur_move])
            negative_evals[i][j]=(engine.get_evaluation()['value'])/100
            negative_positions[i][j]=np.moveaxis(board.lcz_features(), 0, -1)

    high = len([name for name in os.listdir('data') if name.startswith('batch')])
    i = np.random.randint(0, high); a = np.load(f"data/batch{i}.npz")
    idx = np.random.randint(0, a['x'].shape[0]-n_positive)
    positive_positions = a['x'][idx:idx+n_positive]
    positive_evals = a['evals'][idx:idx+n_positive]
    indices = np.arange(chunksize)
    np.random.shuffle(indices)

    if not silent: print(f'positive positions shape is {positive_positions.shape}, positive evals shape is {positive_evals.shape}\
        ')

    y = np.concat([np.zeros(shape=(n_negative, 1)), np.ones(shape=(n_positive, 1))], axis=0)
    all_positions = np.concat([negative_positions, positive_positions], axis=0)
    all_evals = np.concat([negative_evals, positive_evals], axis=0)
    '''Casting to tensors and shuffling data'''
    y = y[indices]
    all_positions = all_positions[indices]
    all_evals = all_evals[indices]
    if all_positions.ndim==6:
        all_positions = np.squeeze(all_positions, axis=0)
    if all_evals.ndim==3:
        all_evals = np.squeeze(all_evals, axis=0)
    if y.ndim==3:
        y = np.squeeze(y, axis=0)
    if save:
        filename_s = f"BinaryClassifierData/{filename}"
        np.savez(file=filename_s, x=all_positions, y = y, evals=all_evals)
        print(f"Successfully saved {filename}")
    else:
        return (all_positions, all_evals, y)

def binary_data_generator(batch_size, generate=False, test=False):
    if generate:
        pos, evals, target = get_binary_chunk(batch_size)
    else:
        if test:
            ar = np.load(f"{DATA_DIR}/test.npz")
        else:
            n = len([name for name in os.listdir(DATA_DIR)])-1
            ar = np.load(f"{DATA_DIR}/batch{np.random.randint(0, n)}.npz")
        pos=ar['x']; target=ar['y']; evals=ar['evals']
    for i in range(min(pos.shape[0], batch_size)):
        yield pos[i], evals[i], target[i]
        
def build_binary_dataset(batch_size, generate = False, test=False):
    dataset = Dataset.from_generator(binary_data_generator, args=[batch_size, generate, test],
                                    output_signature=(
                                        tf.TensorSpec(shape=(5, 8, 8, 112), dtype=tf.int8), #positions
                                        tf.TensorSpec(shape=(5,), dtype=tf.float32), #evals
                                        tf.TensorSpec(shape=(1,), dtype=tf.int8) #targets
                                        ))
    return dataset
    
if __name__=='__main__':
    #get_binary_chunk(chunksize=100, save=True, filename='batch0.npz')

    ds = build_binary_dataset(100,test=True)
    positions, evals, targets = iter(ds.take(1)).next()
    from Model import CNNLSTM
    model = CNNLSTM()
    #print(positions.shape, evals.shape, targets.shape) None, 5, 8, 8 ,112
    #model.training_run(ds)