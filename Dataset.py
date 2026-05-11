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
num_classes = len(targets)
LI_COLS = ['PuzzleId', 'FEN', 'Moves', 'Rating', 'RatingDeviation', 'Popularity', 'NbPlays', 'Themes', 'GameUrl', 'OpeningTags']

def generate_target_vector(tags):
    tag_list = tags.split()
    for i, t in enumerate(targets):
        if t in tag_list:
            vec = np.zeros(num_classes, dtype=np.int8); vec[i] = 1;   return vec
    return np.zeros(num_classes, dtype=np.int8)

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
                        np.ones(chunksize,np.uint8) if binary else np.zeros((chunksize,num_classes),bool))
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

def negative_data_generator(n_instances=1000, chunksize=100, binary=True):
    from stockfish import Stockfish
    eng = Stockfish(path=path_to_binary, depth=1)
    for _ in range(n_instances//chunksize):
        fs, ms = get_game_fens(chunksize)
        p, e = np.zeros((chunksize,5,8,8,112)), np.zeros((chunksize,5))
        if not binary:
            target = np.zeros(shape=(chunksize, num_classes))
            target[:, -1]=1 #No class is true
        else:
            target = np.zeros(chunksize)
        for i in range(len(fs)):
            eng.set_fen_position(fs[i]); b = LeelaBoard(fen=fs[i])
            for j, m in enumerate(ms[i]):
                b.push_uci(m.uci()); eng.make_moves_from_current_position([m.uci()])
                e[i][j] = eng.get_evaluation()['value']/100; p[i][j] = np.moveaxis(b.lcz_features(),0,-1)
        yield p, e, target

def process_df(df, ts=None):
    '''Filtrates df. Creates mask that samples instances with 2<=len(moves)<=5, then randomly changes it by 20%
        returns ts rows, len(df) by default'''
    ts = ts or len(df)
    m = df['Moves'].str.split(' ').str.len().between(2,4)  # Align with `if nl>=5 or nl==1`
    
    # Check that Themes has at least one valid target class
    m &= df['Themes'].str.split().apply(lambda themes: any(t in target_lichess_classes for t in themes))
    
    nf = int(len(df)*0.2)
    if nf>0: m.iloc[np.random.choice(len(df), min(nf,len(df)), replace=False)] ^= True
    d = df[m].copy()
    return d.sample(n=ts, replace=True) if len(d)<ts else d

def generate_precomputed_data(n_batches=1, chunksize=1000, class_weight=0.1, binary:bool=True, filename=None):
    DATA_DIR = BINARY_DATA_DIR if binary else MULTICLASS_DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    files = glob.glob('lichess_db_puzzle/part_*.csv')
    indices = [int(s[s.find('batch')+5:s.find('.')]) for s in glob.glob(f'{DATA_DIR}/batch*.npz')]
    m = max(indices) if len(indices)!=0 else 0
    for b in range(n_batches):
        df = pd.read_csv(np.random.choice(files), names=LI_COLS)
        df = process_df(df)
        idx = np.random.permutation(chunksize)
        if binary:
            np_, ne_ = int(class_weight*chunksize), chunksize-int(class_weight*chunksize)
            px,pe,pt = next(positive_batch_generator(df, n_instances=np_, chunksize=np_, binary=binary))
            nx,ne,nt = next(negative_data_generator(n_instances=ne_, chunksize=ne_, binary=binary))
            pos = np.concatenate([px,nx])[idx]; evals = np.concatenate([pe,ne])[idx]; 
            targ = np.concatenate([pt,nt]).astype(np.int8)[idx]
        else:
            x, e, t = next(positive_batch_generator(df, n_instances=chunksize, chunksize=chunksize, binary=False))
            pos = x[idx]; evals = e[idx]; targ=t[idx]

        np.savez(f'{DATA_DIR}/{f'batch{m+b+1}.npz' if filename is None else filename}', x=pos, 
                 evals=evals, y=targ)




def get_chunk(n_instances=10_000, class_weight=0.1, binary=True, test=False):
    DATA_DIR = BINARY_DATA_DIR if binary else MULTICLASS_DATA_DIR
    fs = sorted(glob.glob(f'{DATA_DIR}/test.npz' if test else f'{DATA_DIR}/batch*.npz'))
    if not fs: raise FileNotFoundError("Run generate_precomputed_data() first")
    
    d = np.load(fs[0], allow_pickle=True, mmap_mode='r')
    x, e, y = d['x'][:n_instances+1], d['evals'][:n_instances+1], d['y'][:n_instances+1]
    
    if binary:
        pos, neg = np.where(y==1)[0], np.where(y==0)[0]
        n_pos, n_neg = int(n_instances*class_weight), n_instances-int(n_instances*class_weight)
        idx = np.concatenate([np.random.choice(pos, n_pos, replace=True), np.random.choice(neg, n_neg, replace=True)])
    else:
        idx = np.arange(len(y))
        
    np.random.shuffle(idx)
    for i in idx[:n_instances]:
        yield x[i], e[i], ([y[i]] if binary else y[i])

def build_binary_dataset(n_instances=10_000, class_weight=class_weight, test=False):
    return Dataset.from_generator(get_chunk, args=[n_instances, class_weight, True, test],
        output_signature=(tf.TensorSpec(shape=(5,8,8,112), dtype=tf.int8), 
                          tf.TensorSpec(shape=(5,), dtype=tf.float32), 
                          tf.TensorSpec(shape=(1,), dtype=tf.int8)))


def build_multiclass_dataset(n_instances=10_000, class_weight=class_weight, test=False):
    return Dataset.from_generator(get_chunk, args=[n_instances, class_weight, False, test],
        output_signature=(tf.TensorSpec(shape=(5,8,8,112), dtype=tf.int8), 
                          tf.TensorSpec(shape=(5,), dtype=tf.float32),
                          tf.TensorSpec(shape=(num_classes,), dtype=tf.uint8)))

if __name__ == '__main__':
    generate_precomputed_data(n_batches=1, chunksize=50, binary=True, filename='test.npz')
    '''
    print("🧪 Running Dataset Tests...")
    
    # Generate minimal test data
    binary_gen = get_chunk(n_instances=100)
    samples = [next(binary_gen) for _ in range(10)]
    x = np.stack([s[0] for s in samples])
    e = np.stack([s[1] for s in samples])
    y = np.stack([s[2] for s in samples])
    print('binary done')
    
    multiclass_gen = get_chunk(n_instances=100, binary=False)
    samples = [next(multiclass_gen) for _ in range(10)]
    x = np.stack([s[0] for s in samples])
    e = np.stack([s[1] for s in samples])
    y = np.stack([s[2] for s in samples])
    print('multiclass done, building bin_ds')
    
    # Test dataset pipeline
    bin_ds = build_binary_dataset(n_instances=10, test=False).batch(10)
    print('initialized, accessing first')
    pos, evl, tgt = next(iter(bin_ds))
    assert pos.shape == (10, 5, 8, 8, 112) and tgt.shape == (10,1), \
        f"Pos or target shape differs from target {pos.shape}\t{tgt.shape}"
    print('bin_ds built, building mult_ds')
    
    mult_ds = build_multiclass_dataset(n_instances=10, test=False).batch(10)
    mlt_pos, mlt_evl, mlt_tgt = next(iter(mult_ds))
    assert mlt_pos.shape == (10, 5, 8, 8, 112) and mlt_tgt.shape == (10,num_classes), \
        f"Pos or target shape differs from target {mlt_pos.shape}\t{mlt_tgt.shape}"
    print('mult_ds built, initializing model')
    
    # Test model forward pass
    from Model import CNNLSTM
    model = CNNLSTM()
    preds = model.binary_call((pos, evl))
    mtl_preds = model((mlt_pos, mlt_evl))
    
    assert preds.shape == (10, 2), f"Binary pred shape mismatch: {preds.shape}"
    assert mtl_preds['multiclass'].shape == (10, num_classes), \
        f"Multiclass pred shape mismatch: {mtl_preds['multiclass'].shape}"
    assert 0 <= tf.reduce_min(preds) <= tf.reduce_max(preds) <= 1, "Probas out of [0,1]"
    
    # ⚠️ Loss/metric tests moved to unittest.ipynb to avoid circular import with Perfomance.py
    print(f"✅ Dataset pipeline: OK | Model forward: OK")
    print(f"   Binary pred shape: {preds.shape} | Multiclass pred shape: {mtl_preds['multiclass'].shape}")
    print(f"   Run unittest.ipynb for loss/metric validation")'''