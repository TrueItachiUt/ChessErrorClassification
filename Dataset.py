'''This file has all information about current dataset
    Not to be confused with Dataset.ipynb, where all experiments and generation is run'''


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