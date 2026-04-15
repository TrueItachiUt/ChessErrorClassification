import os

path_col = '/content/drive/Shareddrives/Shared_drive1/ChessErrorClassification/'
IS_COLAB = os.path.exists('/content/drive/Shareddrives')
IS_KAGGLE = os.path.exists('/kaggle/input')
IS_PROJECT = not (IS_COLAB or IS_KAGGLE)

DATA_DIR = 'BinaryClassifierData'
MODEL_DIR = 'models'
CHECKPOINT_DIR = './checkpoints'

MODEL_FILE_NAME = 'tf_model_19x256.keras'
CHECKPOINT_FILE_NAME = 'last.weights.h5'

if IS_COLAB:
    DATA_DIR = f'{path_col}Data'
    MODEL_DIR = f'{path_col}Models'
    CHECKPOINT_DIR = f'{path_col}Models'