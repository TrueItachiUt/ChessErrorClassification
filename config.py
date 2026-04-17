import os
from dotenv import load_dotenv
load_dotenv()

path_col = '/content/drive/Shareddrives/Shared_drive1/ChessErrorClassification/'
path_kaggle = '/kaggle/input'
var = os.getenv('ENV')
IS_COLAB = IS_KAGGLE = False
if var=='colab':
    IS_COLAB = True
elif var=='kaggle':
    IS_KAGGLE = True
elif var=='local': 
    IS_PROJECT = True
else:
    raise ValueError(f"Invalid environmental variable information about environment.\
                        Valid options are 'colab', 'kaggle', 'local', got {var}")

DATA_DIR = 'BinaryClassifierData'
MODEL_DIR = 'models'
CHECKPOINT_DIR = './checkpoints'

MODEL_FILE_NAME = 'tf_model_19x256.keras'
CHECKPOINT_FILE_NAME = 'last.weights.h5'

if IS_COLAB:
    DATA_DIR = f'{path_col}Data'
    MODEL_DIR = f'{path_col}Models'
    CHECKPOINT_DIR = f'{path_col}Models'

if IS_KAGGLE:
    DATA_DIR = f'{path_kaggle}/datasets/itachiut/binaryclassifierdata'
    MODEL_DIR = f'{path_kaggle}/itachiut/cnnlstm/tensorflow2/default/1'
    CHECKPOINT_DIR = MODEL_DIR