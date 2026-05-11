import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Bidirectional
import keras
from tqdm import tqdm
from IPython import display
import matplotlib.pyplot as plt
from typing import Union
from Perfomance import *
from config import *
from Dataset import num_classes,build_binary_dataset



path_to_checkpoint = f'{CHECKPOINT_DIR}/{CHECKPOINT_FILE_NAME}'
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
            tf.keras.layers.Dense(units=self.n_classes, activation='softmax', name='multiclass_head_dense3')
        ], name='multiclass_classifier_head')
        

        self.CNN.build(input_shape=(None, 8, 8, 112))
        self.norm_1.build(input_shape=(None, 5, 16)); self.norm_2.build(input_shape=(None, n_lstm_blocks*2))
        self.lstm.build(input_shape=(None, 5, 24))
        self.binary_classifier_head.build(input_shape=(None, n_lstm_blocks*2))
        self.multiclass_head.build(input_shape=(None, n_lstm_blocks*2))
        
        self.build(input_shape=(None, 8, 8, 112))

        if os.path.exists(path_to_checkpoint):
            self.load_weights(path_to_checkpoint, skip_mismatch=True)

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


    def training_run(self, ds: tf.data.Dataset, batch_size=20, binary=True):
        '''Runs training across entire ds and saves checkpoint. Visualizes progress after finishing'''
        if os.path.exists(path_to_checkpoint):
            self.load_weights(path_to_checkpoint)
        else:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        self.only_bin = binary
        if binary:
            trainable = self._core() + self.binary_classifier_head.trainable_variables
            opt = self.binary_optimizer
            loss_fn = binary_loss_fn
            metric = BinaryAccuracyMetric()
            get_preds = lambda x: self.binary_call(x)
        else:
            trainable = self._core() + self.multiclass_head.trainable_variables
            opt = self.multiclass_optimizer
            loss_fn = multiclass_loss_fn
            metric = AccuracyMetric()
            get_preds = lambda x: self(x)['multiclass']

        met_vals, losses = [], []
        for batch, (positions, evals, targets) in enumerate(ds.batch(batch_size)):
            with tf.GradientTape() as tape:
                preds = get_preds((positions, evals))
                loss = loss_fn(targets, preds)
            grads = tape.gradient(loss, trainable)
            opt.apply_gradients(zip(grads, trainable))
            metric.update_state(targets, preds)
            met_vals.append(metric.result())
            losses.append(loss.numpy())
            if batch%5==0:
                print(f'Batch {batch} | Loss {loss.numpy()} | {"Balanced Accuracy" if binary else "Accuracy"} {metric.result()}')

        fig = plt.figure(figsize=(10, 6))
        batches = range(1, len(met_vals) + 1)
        plt.plot(batches, met_vals, label='Accuracy', color='yellow')
        plt.plot(batches, losses, label='Loss', color='green')
        plt.xlabel('Batch'); plt.xticks(batches); plt.ylim(0,5)
        plt.legend()
        display.display(fig); plt.close(fig)
        self.save()

    def save(self):
        path_to_checkpoint = f'{CHECKPOINT_DIR}/{CHECKPOINT_FILE_NAME}'
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
    
    def inspect_binary_predicting(self, data: Union[tf.data.Dataset, tuple]):
        if isinstance(data, tf.data.Dataset):
            pos, ev, tar = zip(*data.as_numpy_iterator())  # Fixed 'ds' -> 'data'
            positions = np.stack(pos, axis=0)
            evals = np.stack(ev, axis=0)
            targets = np.stack(tar, axis=0)
        elif isinstance(data, tuple):
            try:
                positions, evals, targets = data
            except Exception as e:
                print(f"Failed to unpack data: ensure you provided positions, evals, targets \n\n {e}")
                return
        else:
            print(f"Unknown type of data {type(data)}")
            return

        mask = (targets == 1).flatten()
        true_positions = positions[mask]; true_evals = evals[mask]; true_targets = targets[mask]

        # Add randomness: include ~25% negatives to the positive set
        n_neg = max(1, len(true_positions) // 4)
        neg_idx = np.random.choice(np.where(~mask)[0], size=min(n_neg, np.sum(~mask)), replace=False)
        mask[neg_idx] = True

        # Shuffle combined indices
        sampled_idx = np.where(mask)[0]
        np.random.shuffle(sampled_idx)
        positions = positions[sampled_idx]
        evals = evals[sampled_idx]
        targets = targets[sampled_idx]

        with tf.GradientTape() as tape:

            preds = self.binary_call((positions, evals))
            loss = binary_loss_fn(targets, preds)
        
        grads = tape.gradient(loss, self._core() + self.binary_classifier_head.trainable_variables)

        grad_norm = tf.linalg.global_norm(grads)
        print(f"Prediction probas for class 0 are {preds[:, 0].numpy()}\n\n targets are {targets}")
        print(f"Loss is {loss}, global gradient norm is {grad_norm}")
        if grad_norm>=10.0:
            layer_name = "LSTM1"  # or layer index, e.g., 2
            for i, var in enumerate(self._core() + self.binary_classifier_head.trainable_variables):
                if layer_name in var.name:
                    layer_grads = grads[i]
                    print(f"{var.name} grad norm: {tf.linalg.global_norm([layer_grads]):.4f}")
                    break
    def evaluate(self, data, batch_size=20):
        '''Evaluates model on dataset. Prints and returns metrics.'''
        bin_loss_met = LossMetric()
        bin_acc = BinaryAccuracyMetric()
        bin_auc = BinaryAUCMetric()
        
        multi_loss_met = LossMetric()
        multi_acc = AccuracyMetric()
        
        for pos, ev, tgt in data.batch(batch_size):
            pos, ev, tgt = tf.cast(pos, tf.float32), tf.cast(ev, tf.float32), tf.convert_to_tensor(tgt)
            
            preds_dict = self((pos, ev))
            
            # Handle binary-only mode or direct tensor return
            if self.only_bin or not isinstance(preds_dict, dict):
                bin_preds = preds_dict
                if len(tgt.shape) == 1: tgt = tf.expand_dims(tgt, -1)
                
                b_loss = binary_loss_fn(tgt, bin_preds)
                bin_loss_met.update_state(b_loss)
                bin_acc.update_state(tgt, bin_preds)
                bin_auc.update_state(tgt, bin_preds)
            else:
                # Route metrics based on target shape
                if tgt.shape[-1] == 1 or (len(tgt.shape) == 1):
                    bin_preds = preds_dict['binary']
                    if len(tgt.shape) == 1: tgt = tf.expand_dims(tgt, -1)
                    
                    b_loss = binary_loss_fn(tgt, bin_preds)
                    bin_loss_met.update_state(b_loss)
                    bin_acc.update_state(tgt, bin_preds)
                    bin_auc.update_state(tgt, bin_preds)
                else:
                    multi_preds = preds_dict['multiclass']
                    m_loss = multiclass_loss_fn(tgt, multi_preds)
                    multi_loss_met.update_state(m_loss)
                    multi_acc.update_state(tgt, multi_preds)
    
        mets = [bin_loss_met, bin_acc, bin_auc, multi_loss_met, multi_acc]
        bin_loss_met_v, bin_acc_v, bin_auc_v, multi_loss_met_v, multi_acc_v = [float(met.result()) for met in mets]
        print(f"Binary   | Loss: {bin_loss_met_v:.4f} | Acc: {bin_acc_v:.3f} | AUC: {bin_auc_v:.3f}")
        print(f"Multiclass| Loss: {multi_loss_met_v:.4f} | Acc: {multi_acc_v:.3f}")
        
        return {
            'binary': {'loss': bin_loss_met_v, 'acc': bin_acc_v, 'auc': bin_auc_v},
            'multiclass': {'loss': multi_loss_met_v, 'acc': multi_acc_v}
        }

    
if __name__=='__main__':
    model = CNNLSTM()
    model([np.random.rand(5,8,8,112), np.random.rand(5)])
    #model.summary()
    ar = np.load(f'{BINARY_DATA_DIR}/test.npz', mmap_mode='r')
    positions = ar['x']; evals = ar['evals'].astype(np.float32); target = ar['y']
    ds = build_binary_dataset(n_instances=200)
    model.evaluate(ds)
    #print(positions.shape, evals.shape, target.shape)
    #model.inspect_binary_predicting(ds)
