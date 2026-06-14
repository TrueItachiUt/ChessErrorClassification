import os
from dotenv import load_dotenv
load_dotenv()

var = os.getenv('ENV')
IS_COLAB = IS_KAGGLE = IS_PROJECT = False
if var=='colab':
    IS_COLAB = True
elif var=='kaggle':
    IS_KAGGLE = True
elif var=='local': 
    IS_PROJECT = True
else:
    raise ValueError(f"Invalid environmental variable information about environment.\
                        Valid options are 'colab', 'kaggle', 'local', got {var}")
                        
PUBLIC_DATA_FOLDER_ID = "1eegVg9K5tn4KqDwbuMyUgeh_lyxGjVTl"
FILES_COUNT = 50
BINARY_DATA_DIR = 'BinaryClassifierData'
MULTICLASS_DATA_DIR = 'data'
MODEL_DIR = 'models'
CHECKPOINT_DIR = './checkpoints'

MODEL_FILE_NAME = 'tf_model_19x256.keras'
CHECKPOINT_FILE_NAME = 'last.weights.h5'

if IS_COLAB:
    BASE_COLAB_PATH = '/content/ChessErrorClassification'
    BINARY_DATA_DIR = f'{BASE_COLAB_PATH}/Data/Binary'
    MULTICLASS_DATA_DIR = f'{BASE_COLAB_PATH}/Data/Multiclass'
    MODEL_DIR = f'{BASE_COLAB_PATH}/Models'
    CHECKPOINT_DIR = f'{BASE_COLAB_PATH}/Models'

if IS_KAGGLE:
    path_kaggle = '/kaggle/input'
    DATA_DIR = f'{path_kaggle}/datasets/itachiut/binaryclassifierdata'
    MODEL_DIR = f'{path_kaggle}/itachiut/cnnlstm/tensorflow2/default/1'
    CHECKPOINT_DIR = MODEL_DIR
    
PATH_TO_STOCKFISH = 'stockfish/stockfish-ubuntu-x86-64-avx2'