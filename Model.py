import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Bidirectional
import keras
from tqdm import tqdm
from IPython import display
import matplotlib.pyplot as plt
from Perfomance import *
from config import *
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
        path = f'{MODEL_DIR}/{MODEL_FILE_NAME}'

        if CNN is None:
            self.CNN = tf.keras.models.load_model(path)
        else:
            self.CNN = CNN
        self.CNN.trainable=False
        
        self.norm_1 = tf.keras.layers.LayerNormalization(name='PreLSTM_layer_norm')

        self.lstm = tf.keras.models.Sequential([
            Bidirectional(tf.keras.layers.LSTM(n_lstm_blocks, return_sequences=True, name='LSTM1')),
            Bidirectional(tf.keras.layers.LSTM(n_lstm_blocks), name='LSTM2')
        ], name='LSTM')

        self.norm_2 = tf.keras.layers.LayerNormalization(name='PostLSTM_layer_norm')

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
        self.norm_1.build(input_shape=(None, 5, 16)); self.norm_2.build(input_shape=(None, n_lstm_blocks*2))
        self.lstm.build(input_shape=(None, 5, 24))
        self.binary_classifier_head.build(input_shape=(None, n_lstm_blocks*2))
        self.multiclass_head.build(input_shape=(None, n_lstm_blocks*2))
        
        self.build(input_shape=(None, 8, 8, 112))
        p_to_ch = f'{CHECKPOINT_DIR}/{CHECKPOINT_FILE_NAME}'
        if os.path.exists(p_to_ch):
            self.load_weights(p_to_ch)

    def _process_CNN(self, inputs):
        """
        Обрабатывает батч позиций через CNN.
        Преобразует 5D тензор (batch, frames, H, W, D) в 3D (batch, frames, features).
        Использует векторизацию через reshape для ускорения.
        """
        inputs = tf.cast(inputs, tf.float32)
        if inputs.ndim==6:
            inputs = tf.squeeze(inputs, axis=[0])
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

        evals = tf.cast(tf.expand_dims(evals, axis=2), tf.float32)
        evals = tf.tile(evals, [1,1,8])
        #batch_size, 5, 8 : every scalar 8 times at axis 1
        assert len(evals.shape)==3, f"Eval's shape is {evals.shape}"
        # Паддинг до фиксированной длины 5
        if vects.shape[1] < 5:
            pad_frames = 5 - vects.shape[1]
            paddings = tf.constant([[0, 0], [0, pad_frames], [0, 0]])
            vects = tf.pad(vects, paddings)
        #Layer Normalization
        vects = self.norm_1(vects)

        #We are less interested in opponent's moves
        mask = tf.constant([1.0, 0.1, 1, 0.1, 1], dtype=tf.float32)
        mask = tf.reshape(mask, [1, 5, 1])
        vects = vects * mask
        return vects, evals
    def _core(self):
        return self.CNN.trainable_variables+self.lstm.trainable_variables
    def call(self, inputs):
        """
        Прямой проход модели.
        inputs: список [positions, evals]
        positions: np.array формы (batch, frames, H, W, D)
        evals: np.array формы (batch, frames,)
        Возвращает словарь с тензорами вероятностей для обучения.
        """
        if self.only_bin:
            return self.binary_call(inputs)

        positions, evals = inputs
        vects = self._process_CNN(positions)

        
        vects, evals = self._prepare_data(vects, evals)
        #print(f"std of vectors is {np.std(vects, axis=None)}, mean is {np.mean(vects, axis=None)}")
        rnn_input = tf.concat([vects, evals], axis=2)

        after_rnn = self.lstm(rnn_input)
        norm_after_rnn = self.norm_2(after_rnn)
        binary_probas = self.binary_classifier_head(norm_after_rnn)
        #print(f"std of rnn output is {np.std(after_rnn)}, mean is {np.mean(after_rnn)}")
        multiclass_probas = self.multiclass_head(norm_after_rnn)

        return {
            'binary': binary_probas,
            'multiclass': multiclass_probas
        }

    def binary_call(self, inputs):
            positions, evals = inputs
            vects = self._process_CNN(positions)

            
            vects, evals = self._prepare_data(vects, evals)
            rnn_input = tf.concat([vects, evals], axis=2)
            after_rnn = self.norm_2(self.lstm(rnn_input))
            return self.binary_classifier_head(after_rnn)


    def training_run(self, ds: tf.data.Dataset, batch_size=20):
        '''Runs training across entire ds and saves checkpoint. Visualizes progress after finishing'''
        path_to_checkpoint = f'{CHECKPOINT_DIR}/{CHECKPOINT_FILE_NAME}'
        if os.path.exists(path_to_checkpoint):
            self.load_weights(path_to_checkpoint)
        else:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self.only_bin = True
        binary_trainable = self._core() + self.binary_classifier_head.trainable_variables
        bacm = BinaryAccuracyMetric()
        bin_acc = []
        losses = []
        
        for batch, (positions, evals, targets) in enumerate(ds.batch(batch_size)):
            with tf.GradientTape() as tape:
                preds = self.binary_call((positions,evals))
                loss = binary_loss_fn(targets, preds)
            print(loss)
            grads = tape.gradient(loss, binary_trainable)
            self.binary_optimizer.apply_gradients(zip(grads, binary_trainable))
            
            bacm.update_state(targets, preds)
            bin_acc.append(bacm.result())
            losses.append(loss.numpy())
            if (batch%5==0):
                print(f'Batch {batch} | Loss {loss.numpy()} | Balanced Accuracy {bacm.result()}')

        fig = plt.figure(figsize=(10, 6))
        batches = range(1, len(bin_acc) + 1)
        plt.plot(batches, bin_acc, label='Accuracy', color='yellow')
        plt.plot(batches, losses, label='Loss', color='green')
        plt.xlabel('Batch')
        plt.xticks(batches)
        plt.legend()
        display.display(fig); plt.close(fig)
        self.save_weights(path_to_checkpoint)

    def inspect(self, inputs):
        positions, evals = inputs
        print(f"Position dtype {positions.dtype} Evals mean {np.mean(evals)}, std {np.std(evals)}")
        vects = self._process_CNN(positions)
        print(f"Mean of CNN output with 2 Dense layers {np.mean(vects)}, std {np.std(vects)}, shape {tf.shape(vects)}")
        vects, evals = self._prepare_data(vects, evals)
        print(f'Mean of prepared to lstm data {np.mean(vects)}, shape {tf.shape(vects)}')
        rnn_input = tf.concat([vects, evals], axis=2)
        after_rnn = self.lstm(rnn_input)
        norm_after_rnn = self.norm_2(after_rnn)
        print(f"Shape of after rnn {tf.shape(after_rnn)}, mean {np.mean(after_rnn)}, std {np.std(after_rnn)}")
        print(f"Shape of noprmalized after rnn {tf.shape(norm_after_rnn)}, mean {np.mean(norm_after_rnn)}, std {np.std(norm_after_rnn)}")
        binary_head_out = self.binary_classifier_head(norm_after_rnn)
        print(f"After binary classifier shape {tf.shape(binary_head_out)}, mean at axis 0 {np.mean(binary_head_out, axis=0)}")
        print(f"Binary values output: {binary_head_out}")

    
if __name__=='__main__':
    model = CNNLSTM()
    model([np.random.rand(5,8,8,112), np.random.rand(5)])
    #model.summary()
    ar = np.load(f'{DATA_DIR}/test.npz')
    positions = ar['x']; evals = ar['evals'].astype(np.float32); target = ar['y']
    #print(positions.shape, evals.shape, target.shape)
    with tf.GradientTape() as tape:
        pred = model.binary_call((positions, evals))
        loss = binary_loss_fn(target, pred)
    print(loss)
    model.inspect((positions[:5], evals[:5]))
