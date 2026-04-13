import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Bidirectional
import keras
from Perfomance import *
from Dataset import num_classes

#@keras.utils.register_keras_serializable()
class CNNLSTM(tf.keras.Model):
    """
    Модель CNN-LSTM для классификации последовательностей позиций.
    Принимает на вход позиции и оценки, возвращает вероятности классов.
    """
    def __init__(self, CNN=None, n_lstm_blocks=32, n_lstm_layers=2, bidirectional=True, window_length=5, only_bin=False, 
                binary_optimizer=None, multiclass_optimizer=None):
        self.window_length = window_length
        self.only_bin = only_bin
        self.binary_optimizer=tf.keras.optimizers.SGD(momentum=0.9) if not binary_optimizer else binary_optimizer
        self.multiclass_optimizer=tf.keras.optimizers.SGD(momentum=0.9) if not binary_optimizer and not only_bin else binary_optimizer
        super(CNNLSTM, self).__init__()
        self.n_classes = num_classes
        path = 'models/tf_model_19x256.keras'

        if CNN is None:
            self.CNN = tf.keras.models.load_model(path)
        else:
            self.CNN = CNN
        self.CNN.trainable=False
        
        
        self.lstm = tf.keras.models.Sequential([
            Bidirectional(tf.keras.layers.LSTM(n_lstm_blocks, return_sequences=True, name='LSTM1')),
            Bidirectional(tf.keras.layers.LSTM(n_lstm_blocks), name='LSTM2')
        ], name='LSTM')


        self.binary_classifier_head = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=16, activation='tanh', name='binary_head_dense1'),
            tf.keras.layers.Dense(units=2, activation='softmax', name='binary_head_dense2')
        ], name='binary_classifier_head')

        self.multiclass_head = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=32, activation='tanh', name='multiclass_head_dense1'),
            tf.keras.layers.Dense(units=16, activation='tanh', name='multiclass_head_dense2'),
            tf.keras.layers.Dense(units=self.n_classes-1, activation='softmax', name='multiclass_head_dense3')
        ], name='multiclass_classifier_head')

        self.CNN.build(input_shape=(None, 8, 8, 112))
        self.lstm.build(input_shape=(None, 5, 24))
        self.binary_classifier_head.build(input_shape=(None, n_lstm_blocks*2))
        self.multiclass_head.build(input_shape=(None, n_lstm_blocks*2))


    def _process_CNN(self, inputs):
        """
        Обрабатывает батч позиций через CNN.
        Преобразует 5D тензор (batch, frames, H, W, D) в 3D (batch, frames, features).
        Использует векторизацию через reshape для ускорения.
        """
        inputs = tf.convert_to_tensor(inputs)
        if len(inputs.shape) == 4:
            inputs = tf.expand_dims(inputs, axis=0)

        shape = tf.shape(inputs)
        batch_size, n_frames = shape[0], shape[1]
        
        # Объединяем батч и время для пакетной обработки CNN
        inputs_reshaped = tf.reshape(inputs, [-1, shape[2], shape[3], shape[4]])
        cnn_out = self.CNN(inputs_reshaped)
        feature_dim = tf.shape(cnn_out)[-1]

        # Возвращаем размерность времени обратно
        return tf.reshape(cnn_out, [batch_size, n_frames, feature_dim])


    def _prepare_data(self, vects, evals):
        '''
           Prepares vectors of CNN output and evaluations for rnn
           Methods implemented there:
            - Batching
            - Setting length of input data to window length - clipping/padding
            - Extending evals along 1 dimension - repeating every scalar 8 times
            Takes - vects after CNN (tf.tensor), evals (unbatched or batched) (numpy array)
            Returns - prepared vects and evals tf tensors
        '''
        if len(evals.shape)==1:
            evals = tf.expand_dims(evals, axis=0) #batch_size, n_frames


        if evals.shape[1]<self.window_length:
            evals = tf.pad(evals, [[0,0], [0, 5-evals.shape[1]]])

        else:
            evals = evals[:, :self.window_length]
        #batch_size, 5

        evals = tf.expand_dims(evals, axis=2)
        evals = tf.tile(evals, [1,1,8])
        #batch_size, 5, 8 : every scalar 8 times at axis 1
        assert len(evals.shape)==3, f"Eval's shape is {evals.shape}"
        # Паддинг до фиксированной длины 5
        if vects.shape[1] < 5:
            pad_frames = 5 - vects.shape[1]
            paddings = tf.constant([[0, 0], [0, pad_frames], [0, 0]])
            vects = tf.pad(vects, paddings)
            
        #We are less interested in opponent's moves
        mask = tf.constant([1.0, 0.1, 1, 0.1, 1], dtype=tf.float32)
        mask = tf.reshape(mask, [1, 5, 1])
        vects = vects * mask
        return vects, evals

    def call(self, inputs):
        """
        Прямой проход модели.
        inputs: список [positions, evals]
        positions: np.array формы (batch, frames, H, W, D)
        evals: np.array формы (batch, frames,)
        Возвращает словарь с тензорами вероятностей для обучения.
        """
        positions, evals = inputs
        vects = self._process_CNN(positions)

        
        vects, evals = self._prepare_data(vects, evals)
        #print(f"std of vectors is {np.std(vects, axis=None)}, mean is {np.mean(vects, axis=None)}")
        rnn_input = tf.concat([vects, evals], axis=2)

        after_rnn = self.lstm(rnn_input)
        binary_probas = self.binary_classifier_head(after_rnn)
        #print(f"std of rnn output is {np.std(after_rnn)}, mean is {np.mean(after_rnn)}")
        multiclass_probas = self.multiclass_head(after_rnn)

        return {
            'binary': binary_probas,
            'multiclass': multiclass_probas
        }

    def binary_call(self, inputs):
            positions, evals = inputs
            vects = self._process_CNN(positions)

            
            vects, evals = self._prepare_data(vects, evals)
            rnn_input = tf.concat([vects, evals], axis=2)
            after_rnn = self.lstm(rnn_input)
            return self.binary_classifier_head(after_rnn)

    def binary_training_step(self, inputs, target):

            with tf.GradientTape() as tape:
                preds = self.binary_call(inputs)
                loss = binary_loss_fn(target, preds)
            for metric in self.metrics:
                if metric.name=='loss':
                    metric.update_state(loss)
                else:
                    metric.update_state(target, preds)

            binary_trainable = self.lstm.trainable_variables+self.binary_classifier_head.trainable_variables+self.CNN.trainable_variables
            grads = tape.gradient(loss, binary_trainable)
            self.binary_optimizer.apply_gradients(zip(grads, binary_trainable))

    def training_run(self, ds: tf.data.Dataset, batch_size=20):
        '''Runs an epoch on entire ds and saves checkpoint'''
        self.only_bin = True
        for (positions, evals, targets) in tqdm(ds.batch(batch_size)):
            self.binary_training_step((positions, evals), targets)
    
if __name__=='__main__':
    model = CNNLSTM()
    model([np.random.rand(5,8,8,112), np.random.rand(5)])
    #model.summary()
    ar = np.load('BinaryClassifierData/test.npz')
    positions = ar['x']; evals = ar['evals'].astype(np.float32); target = ar['y']
    #print(positions.shape, evals.shape, target.shape)
    model.binary_training_step((positions, evals), target)
    for metric in model.metrics:
        print(metric.result())