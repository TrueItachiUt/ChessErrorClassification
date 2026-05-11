import tensorflow as tf
from tensorflow.keras.losses import BinaryFocalCrossentropy, CategoricalCrossentropy
import numpy as np
from Dataset import num_classes, class_weight
from tensorflow.keras import ops


class LossMetric(tf.keras.Metric):
    def __init__(self, **kwargs):
        super().__init__(name='loss', **kwargs)
        self.sum = self.add_variable(shape=(), initializer='zeros', name='sum')
        self.cnt = self.add_variable(shape=(), initializer='zeros', name='count')
    def update_state(self, loss):
        self.sum+=loss
        self.cnt+=1
    def result(self):
        return self.sum/self.cnt

    
'''Важно помнить что y_true[:, -1] это no_class столбец, если 1 то удара нет
В предсказаниях y_pred[:, 1] - вероятность того, что удар есть'''
class BinaryAUCMetric(tf.keras.metrics.AUC):
    def __init__(self, name='CustomBinaryAUC', **kwargs):
        super().__init__(name=name, curve='PR', **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_true.shape)!=1 and y_true.shape[1]>1:
            y_true = tf.cast(tf.equal(y_true[:,-1], 0), tf.int8) #Multiclass labels
        if isinstance(y_pred, dict):
            y_pred = y_pred['binary']
        binary_probas = y_pred[:, 1]
        super().update_state(y_true, binary_probas)
    def result(self):
        return super().result()


class BinaryAccuracyMetric(tf.keras.Metric):

    def __init__(self, name='BinaryAccuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp=self.add_variable(shape=(),initializer='zeros',name='tp')
        self.fp=self.add_variable(shape=(),initializer='zeros',name='fp')
        self.fn = self.add_variable(shape=(), initializer='zeros', name='fn')
        self.tn=self.add_variable(shape=(),initializer='zeros',name='tn')


    def update_state(self, y_true, y_pred, sample_weight=None):

        if isinstance(y_pred, dict):
            y_true = ops.equal(y_true[:, -1], 0)
            y_pred = y_pred['binary'] 
    
        # FIX: Squeeze to 1D to prevent (N,) vs (N,1) broadcasting
        y_true_flat = tf.squeeze(y_true)
        y_pred_bool = y_pred[:, 1] > 0.5
        
        # Element-wise boolean comparisons (both 1D now)
        tp_sampl = tf.logical_and(y_pred_bool, tf.cast(y_true_flat, tf.bool))
        tn_sampl = tf.logical_and(tf.logical_not(y_pred_bool), tf.logical_not(tf.cast(y_true_flat, tf.bool)))
        fp_sampl = tf.logical_and(y_pred_bool, tf.logical_not(tf.cast(y_true_flat, tf.bool)))
        fn_sampl = tf.logical_and(tf.logical_not(y_pred_bool), tf.cast(y_true_flat, tf.bool))
        
        self.tp.assign_add(tf.reduce_sum(tf.cast(tp_sampl, self.dtype)))
        self.tn.assign_add(tf.reduce_sum(tf.cast(tn_sampl, self.dtype)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(fp_sampl, self.dtype)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(fn_sampl, self.dtype)))


    def result(self):
        if self.tp == 0 and self.tn == 0: 
            return 0
        TPR = self.tp/(self.tp+self.fn) #Sensivity
        TNR = self.tn/(self.fp+self.tn) #Specitivity
        return (TPR+TNR)/2
    

class AccuracyMetric(tf.keras.metrics.Accuracy):

    def __init__(self, name='CustomAccuracy', **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if isinstance(y_pred, dict): y_pred = y_pred['multiclass']
        n = (y_true.shape)[1]
        if n==num_classes+1:
            #Raw data with no class column
            mask = y_true[:, -1]==0
            y_true = (y_true[mask])[:, :-1]
            y_pred = y_pred[mask]
        elif n==num_classes:
            pass
        else:
            raise ValueError(f"Passed target of unsupported shape : expected either {num_classes} (pure multiclass\
                             one hot) or {num_classes+1} (raw detection data), got {n}")
        super().update_state(y_true, y_pred) 

    def result(self):
        return super().result()


#bfce = BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.9) #Class 1 is 10 times more important than class 0 
bce = tf.keras.losses.BinaryCrossentropy(reduction=None)
cce = CategoricalCrossentropy() #To handle loss


@tf.function
def binary_loss_fn(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred[:, 1], axis=-1)

    y_true_broadcasted = tf.expand_dims(y_true, axis=-1)

    loss_ar = bce(y_true_broadcasted, y_pred)

    weights = tf.where(y_true == 1, (1 / class_weight), 1)
             
    return tf.reduce_mean(loss_ar*weights)

@tf.function
def multiclass_loss_fn(y_true, y_pred):
    '''Loss for multiclass output probas and multiclass one-hot target'''
    if len(tf.shape(y_true)) == 1:
        y_true = tf.expand_dims(y_true, axis=0)
    if len(y_pred.shape) == 1:
        y_pred = tf.expand_dims(y_pred, axis=0)
    tf.debugging.assert_equal(tf.shape(y_true)[1], num_classes, message="Target shape mismatch")
    return cce(y_true, y_pred)

def detection_loss(y_true, y_pred):
    bin_y_true = tf.cast(tf.equal(y_true[:, -1], 0), tf.int8) #In multiclass classification last column is 1 if 
                                                          # 'no class', so we need to inverse it for binary detection
    mult_y_true = y_true[:, :-1]
    return {
        'binary': binary_loss_fn(bin_y_true, y_pred['binary']),
        'multiclass': multiclass_loss_fn(mult_y_true , y_pred['multiclass'])
    }

#def fit(self, X, y, eval_set: tuple = None):
if __name__=='__main__':

    from Model import CNNLSTM
    model = CNNLSTM()
    model.compile(optimizer='SGD', loss={
        'binary': binary_loss_fn,
        'multiclass': multiclass_loss_fn
    }, metrics = {'binary': BinaryAUCMetric(),'multiclass':AccuracyMetric()})
    #print(model.summary())
    batch_size=20
    
    positions = np.random.randn(batch_size,4,8,8,112)
    evals = np.random.randn(batch_size,4)
    target = np.zeros(shape=(batch_size,num_classes))
    for i in range(batch_size):
        if np.random.rand()<=class_weight:
            class_value = num_classes-1
        else:
            class_value = np.random.randint(low=0, high=num_classes-1)
        target[i, class_value]=1
    
    '''ar = np.load("data/batch0.npz")
    positions = ar['x'][:batch_size]
    target = ar['y'][:batch_size]
    
    evals = ar['evals'][:batch_size]
    print(positions.shape, target.shape, evals.shape)
    target = np.append(target, np.zeros((target.shape[0], 1)), axis=1) #For dimensionality match'''
    bin_target = tf.cast(tf.equal(target[:, -1], 0), tf.int8)
    with tf.GradientTape() as tape:
        preds = model((positions, evals))
        binary_loss = binary_loss_fn(bin_target, preds['binary'])
        multiclass_loss = multiclass_loss_fn(target, preds['multiclass'])
        loss = binary_loss+multiclass_loss

    grad = tape.gradient(loss, model.trainable_variables)
    #print(f"\n\n Preds are {preds} \n\n")
    print(multiclass_loss, binary_loss)
    #print(grad)
    
    binary_auc = BinaryAUCMetric()
    binary_auc.update_state(target, preds)
    print(f"Binary AUC is {binary_auc.result()}")
    #model.evaluate()
    binary_acc = BinaryAccuracyMetric()
    binary_acc.update_state(target, preds)
    print(f"Binary accuracy is {binary_acc.result()}")
    if binary_acc.result() is np.nan:
        
        print(f"Binary: \n preds {preds['binary'][:, 1]} \n\n target {bin_target}")

    multiclass_acc = AccuracyMetric()
    multiclass_acc.update_state(target, preds)
    print(f"Class prediction accuracy is {multiclass_acc.result()}")

    #model.evaluate(x=(positions, evals), y={'binary':target, 'multiclass':bin_target},verbose=1)