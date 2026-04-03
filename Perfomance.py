import tensorflow as tf
from tensorflow.keras.losses import BinaryFocalCrossentropy, CategoricalCrossentropy
import numpy as np
from Dataset import num_classes, class_weight
from tensorflow.keras import ops

'''Важно помнить что y_true[:, -1] это no_class столбец, если 1 то удара нет
В предсказаниях y_pred[:, 1] - вероятность того, что удар есть'''
class BinaryAUCMetric(tf.keras.metrics.AUC):
    def __init__(self, name='CustomBinaryAUC', **kwargs):
        super().__init__(name=name, curve='PR', **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.equal(y_true[:,-1], 0), tf.int8)
        if isinstance(y_pred, dict):
            y_pred = y_pred['binary']
        binary_probas = y_pred[:, 1]
        super().update_state(y_true, binary_probas)
    def result(self):
        return super().result()


class BinaryAccuracyMetric(tf.keras.Metric):

    def __init__(self, bin=False, name='BinaryAccuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.bin = bin
        self.tp=self.add_variable(shape=(),initializer='zeros',name='tp')
        self.fp=self.add_variable(shape=(),initializer='zeros',name='fp')
        self.fn = self.add_variable(shape=(), initializer='zeros', name='fn')
        self.tn=self.add_variable(shape=(),initializer='zeros',name='tn')


    def update_state(self, y_true, y_pred):

        if not self.bin:
            y_true = ops.equal(y_true[:, -1], 0) #inverse
            y_pred = y_pred['binary'] 
        assert len(y_true.shape)==1 or y_true.shape[1]==1, f"Expected dense target, got shape {y_true.shape}"
        y_pred = ops.cast(y_pred[:, 1], "bool") #Threshold is 0.5
        tp_sampl = ops.logical_and(
            ops.equal(y_pred, True), ops.equal(y_true, True)
        )
        tn_sampl = ops.logical_and(
            ops.equal(y_pred, False), ops.equal(y_true, False)
        )
        fp_sampl = ops.logical_and(
            ops.equal(y_pred, True), ops.equal(y_true, False)
        )
        fn_sampl = ops.logical_and(
            ops.equal(y_pred, False), ops.equal(y_true, True)
        )
        self.tp.assign(self.tp+ops.sum(ops.cast(tp_sampl, self.dtype)))
        self.tn.assign(self.tn+ops.sum(ops.cast(tn_sampl, self.dtype)))
        self.fp.assign(self.fp+ops.sum(ops.cast(fp_sampl, self.dtype)))
        self.fn.assign(self.fn+ops.sum(ops.cast(fn_sampl, self.dtype)))


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
        mask = y_true[:, -1]==0
        targets_one_hot = (y_true[mask])[:, :-1]
        y_pred = y_pred[mask]
        super().update_state(targets_one_hot, y_pred) 

    def result(self):
        return super().result()


bfce = BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.9) #Class 1 is 10 times more important than class 0 
cce = CategoricalCrossentropy(reduction='none') #To handle loss
@tf.function
def binary_loss_fn(y_true, y_pred):
    '''Loss for binary output probas and multiclass one-hot target'''
    if len(tf.shape(y_true)) == 1:
        y_true = tf.expand_dims(y_true, axis=0)
    y_true = tf.cast(tf.equal(y_true[:, -1], 0), tf.int8) #In multiclass classification last column is 1 if 
                                                          # 'no class', so we need to inverse it for binary detection
    if len(y_pred.shape)==1:
        y_pred = tf.expand_dims(y_pred, axis=0)
    y_pred = y_pred[:, 1]
    return bfce(y_true, y_pred)
@tf.function
def multiclass_loss_fn(y_true, y_pred):
    '''Loss for multiclass output probas and multiclass one-hot target'''
    if len(tf.shape(y_true)) == 1:
        y_true = tf.expand_dims(y_true, axis=0)

    if len(y_pred.shape) == 1:
        y_pred = tf.expand_dims(multiclass_probas, axis=0)

    original_batch_size = tf.shape(y_true)[0]
    
    # Multiclass part
    mask = tf.cast(tf.equal(y_true[:, -1], 0), tf.bool) #Take only instances when last column is zero
    
    multiclass_targets = tf.boolean_mask(y_true[:, :-1], mask)

    

    if tf.shape(y_pred)[0]==original_batch_size:  # If some value was used to fill no class instances or 
                                                             #predicted multiclass for all instances
        y_pred = tf.boolean_mask(y_pred, mask)
    
    #print(f"Mask shape is {tf.shape(mask)}, after masking multiclass probas is {tf.shape(multiclass_probas)}\
        #targets are {tf.shape(multiclass_targets)}")

    per_sample_losses = cce(multiclass_targets, y_pred)
    
    # Суммируем ошибки и делим на ИСХОДНЫЙ размер батча
    # Это сохраняет масштаб градиента стабильным независимо от количества объектов
    return tf.reduce_sum(per_sample_losses) / tf.cast(original_batch_size, tf.float32)

def detection_loss(y_true, y_pred):
    return {
        'binary': binary_loss_fn(y_true, y_pred['binary']),
        'multiclass': multiclass_loss_fn(y_true , y_pred['multiclass'])
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
    
    with tf.GradientTape() as tape:
        preds = model((positions, evals))
        binary_loss = binary_loss_fn(target, preds['binary'])
        multiclass_loss = multiclass_loss_fn(target, preds['multiclass'])
        loss = binary_loss+multiclass_loss

    grad = tape.gradient(loss, model.trainable_variables)
    #print(f"\n\n Preds are {preds} \n\n")
    
    #print(grad)
    
    binary_auc = BinaryAUCMetric()
    binary_auc.update_state(target, preds)
    print(f"Binary AUC is {binary_auc.result()}")
    #model.evaluate()
    binary_acc = BinaryAccuracyMetric()
    binary_acc.update_state(target, preds)
    print(f"Binary accuracy is {binary_acc.result()}")
    multiclass_acc = AccuracyMetric()
    multiclass_acc.update_state(target, preds)
    print(f"Class prediction accuracy is {multiclass_acc.result()}")

    #model.evaluate(x=(positions, evals), y={'binary':target, 'multiclass':target},verbose=1)