import numpy as np, pandas as pd, os, glob, random, chess.pgn, tensorflow as tf
from minimal_lc0_for_research.leela_board import LeelaBoard
from tensorflow.data import Dataset
from config import *

file_number = 5
class_weight=0.1
path_to_binary = 'stockfish/stockfish-ubuntu-x86-64-avx2'
target_lichess_classes = ['exposedKing','sacrifice','hangingPiece','fork','captureTheDefender','pin','quietMove','intermezzo','deflection']
additional_target_classes = ['planlessGame']
targets = target_lichess_classes + additional_target_classes
num_classes = len(targets) + 1
LI_COLS = ['PuzzleId', 'FEN', 'Moves', 'Rating', 'RatingDeviation', 'Popularity', 'NbPlays', 'Themes', 'GameUrl', 'OpeningTags']

def generate_target_vector(tags): return np.array([t in tags.split() for t in targets], dtype=bool)

def get_game_fens(n, pgn='data/lichess_elite_2024-05.pgn'):
    FENs, moves = [], []
    with open(pgn) as f:
        for _ in range(n):
            g = chess.pgn.read_game(f)
            if not g: break
            ml = list(g.mainline_moves())
            if len(ml)<2: continue
            l = min(random.choices([2,3,4,5], weights=[1,1,2,4], k=1)[0], len(ml))
            s = random.randint(0, len(ml)-l); b = g.board()
            for m in ml[:s]: b.push(m)
            FENs.append(b.fen()); moves.append(ml[s:s+l])
    return FENs, moves

def positive_batch_generator(df, n_instances=1000, chunksize=100, binary=True):
    from stockfish import Stockfish
    c, eng = 0, Stockfish(path=path_to_binary, depth=1)
    for _ in range(n_instances//chunksize):
        pos, evs, tgs = (np.zeros((chunksize,5,8,8,112),np.uint8), 
                        np.zeros((chunksize,5)), 
                        np.ones(chunksize,np.uint8) if binary else np.zeros((chunksize,num_classes-1),bool))
        f = 0
        while f<chunksize and c<len(df):
            fen, m, th = df['FEN'].values[c], df['Moves'].values[c], df['Themes'].values[c]; c+=1
            nl = len(m.split(' '))  # FIX: split by space to count moves
            if nl>=5 or nl==1: continue
            b = LeelaBoard(fen=fen); eng.set_fen_position(fen)
            for k in range(nl):
                pos[f][k] = np.moveaxis(b.lcz_features(),0,-1)
                evs[f][k] = eng.get_evaluation()['value']/100
                if k<nl-1: 
                    b.push_uci(m.split()[k]); 
                    eng.make_moves_from_current_position([m.split()[k]])

            if not binary: tgs[f] = generate_target_vector(th)
            f+=1
        yield pos[:f], evs[:f], tgs[:f]

def negative_data_generator(n_instances=1000, chunksize=100):
    from stockfish import Stockfish
    eng = Stockfish(path=path_to_binary, depth=1)
    for _ in range(n_instances//chunksize):
        fs, ms = get_game_fens(chunksize)
        p, e = np.zeros((len(fs),5,8,8,112)), np.zeros((len(fs),5))
        for i in range(len(fs)):
            eng.set_fen_position(fs[i]); b = LeelaBoard(fen=fs[i])
            for j, m in enumerate(ms[i]):
                b.push_uci(m.uci()); eng.make_moves_from_current_position([m.uci()])
                e[i][j] = eng.get_evaluation()['value']/100; p[i][j] = np.moveaxis(b.lcz_features(),0,-1)
        yield p, e, np.zeros(len(fs))

def process_df(df, ts=None):
    '''Filtrates df. Creates mask that samples instances with 2<=len(moves)<=5, then randomly changes it by 20%
        returns ts rows, len(df) by default'''
    ts = ts or len(df)
    m = df['Moves'].str.split(' ').str.len().between(2,4)  # Align with `if nl>=5 or nl==1`
    nf = int(len(df)*0.2)
    if nf>0: m.iloc[np.random.choice(len(df), min(nf,len(df)), replace=False)] ^= True
    d = df[m].copy()
    return d.sample(n=ts, replace=True) if len(d)<ts else d

def generate_precomputed_data(n_batches=5, chunksize=1000, class_weight=0.1):
    os.makedirs(DATA_DIR, exist_ok=True)
    files = glob.glob('lichess_db_puzzle/part_*.csv')
    for b in range(n_batches):
        df = pd.read_csv(np.random.choice(files), names=LI_COLS)
        df = process_df(df)
        np_, ne_ = int(class_weight*chunksize), chunksize-int(class_weight*chunksize)
        px,pe,pt = next(positive_batch_generator(df, np_, np_, True))
        nx,ne,nt = next(negative_data_generator(ne_, ne_))
        idx = np.random.permutation(len(px)+len(nx))
        np.savez(f'{DATA_DIR}/batch{b}.npz', x=np.concatenate([px,nx])[idx], 
                 evals=np.concatenate([pe,ne])[idx], y=np.concatenate([pt,nt]).astype(np.int8)[idx])

def get_binary_chunk(n_instances=10_000, class_weight=0.1, test=False):
    fs = sorted(glob.glob(f'{DATA_DIR}/test.npz' if test else f'{DATA_DIR}/batch*.npz'))
    if not fs: raise FileNotFoundError("Run generate_precomputed_data() first")
    yielded, fid = 0, 0
    while yielded < n_instances:
        d = np.load(fs[fid%len(fs)], allow_pickle=True); fid += 1
        x, e, y = d['x'], d['evals'], d['y']
        pos, neg = np.where(y==1)[0], np.where(y==0)[0]
        n_pos, n_neg = int(n_instances*class_weight), n_instances-int(n_instances*class_weight)
        if len(pos)<n_pos or len(neg)<n_neg: continue
        idx = np.concatenate([np.random.choice(pos,n_pos,replace=False), np.random.choice(neg,n_neg,replace=False)])
        np.random.shuffle(idx)
        for i in idx: yield x[i], e[i], [y[i]]; yielded += 1

def build_binary_dataset(n_instances=10_000, class_weight=class_weight, test=False):
    return Dataset.from_generator(get_binary_chunk, args=[n_instances, class_weight, test],
        output_signature=(tf.TensorSpec(shape=(5,8,8,112), dtype=tf.int8), 
                          tf.TensorSpec(shape=(5,), dtype=tf.float32), 
                          tf.TensorSpec(shape=(1,), dtype=tf.int8)))

if __name__ == '__main__':
    '''generate_precomputed_data(n_batches=3, chunksize=5_000)'''
    print("🧪 Running Dataset Tests...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Generate minimal test data
    
    gen = get_binary_chunk(n_instances=100)
    samples = [next(gen) for _ in range(10)]
    x = np.stack([s[0] for s in samples])      # (50, 5, 8, 8, 112)
    e = np.stack([s[1] for s in samples])      # (50, 5)
    y = np.stack([s[2] for s in samples])      # (50,) or (50,1) depending on signature

    
    # Test dataset pipeline - add .batch() since from_generator yields unbatched
    ds = build_binary_dataset(10, test=False).batch(10)
    pos, evl, tgt = next(iter(ds))
    assert pos.shape == (10, 5, 8, 8, 112) and tgt.shape == (10,1), f"Pos or target  shape differs from target {pos.shape}\t{tgt.shape}"
    
    # Test model forward pass
    from Model import CNNLSTM
    model = CNNLSTM()
    preds = model.binary_call((pos, evl))
    
    assert preds.shape == (10, 2) and 0 <= tf.reduce_min(preds) <= tf.reduce_max(preds) <= 1
    
    # Test loss/metrics
    from Perfomance import binary_loss_fn, BinaryAccuracyMetric, BinaryAUCMetric
    loss = binary_loss_fn(tgt, preds)
    assert not tf.math.is_nan(loss)
    
    acc, auc = BinaryAccuracyMetric(), BinaryAUCMetric()
    acc.update_state(tgt, preds); auc.update_state(tgt, preds)
    
    print(f"✅ Loss:{loss.numpy():.4f} Acc:{acc.result().numpy():.3f} AUC:{auc.result().numpy():.3f}")