import numpy as np
import h5py, os
from six.moves import cPickle
import argparse
import matplotlib.pyplot as plt

import tensorflow as tf
import tfomics
from tfomics import explain, evaluate
import models


#-----------------------------------------------------------------

def load_synthetic_data(file_path):

    with h5py.File(file_path, 'r') as dataset:
        x_train = np.array(dataset['X_train']).astype(np.float32).transpose([0, 2, 1])
        y_train = np.array(dataset['Y_train']).astype(np.float32)
        x_valid = np.array(dataset['X_valid']).astype(np.float32).transpose([0, 2, 1])
        y_valid = np.array(dataset['Y_valid']).astype(np.int32)
        x_test = np.array(dataset['X_test']).astype(np.float32).transpose([0, 2, 1])
        y_test = np.array(dataset['Y_test']).astype(np.int32)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

#-----------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-m", type=str, default=0.05, help="model_name")
parser.add_argument("-p", type=int, default=20, help="pool_size")
parser.add_argument("-a", type=str, default='relu', help="activation")
parser.add_argument("-f", type=int, default=48, help="filters")
parser.add_argument("-t", type=int, default=None, help="trial")
args = parser.parse_args()

model_name = args.m
pool_size = args.p
activation = args.a
trial = args.t
num_filters = args.f

# set paths
results_path = '../results_task1'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# load data
data_path = '../../data'
filepath = os.path.join(data_path, 'synthetic_dataset.h5')
data = load_synthetic_data(filepath)
x_train, y_train, x_valid, y_valid, x_test, y_test = data
N, L, A = x_train.shape
num_labels = y_train.shape[1]

# build model
if model_name == 'CNN_ATT':
    model = models.CNN_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                           num_filters=num_filters, dense_units=96, heads=8, key_size=48)

elif model_name == 'CNN_LSTM':
    model = models.CNN_LSTM(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                            num_filters=num_filters, lstm_units=48, dense_units=96)

elif model_name == 'CNN_LSTM_ATT':
    model = models.CNN_LSTM_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                num_filters=num_filters, lstm_units=48, dense_units=96, heads=8, key_size=96)
elif model_name == 'CNN_LSTM_TRANS1':
    model = models.CNN_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=1, heads=8, key_size=48, dense_units=96)
elif model_name == 'CNN_LSTM_TRANS2':
    model = models.CNN_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=2, heads=8, key_size=48, dense_units=96)
elif model_name == 'CNN_LSTM_TRANS4':
    model = models.CNN_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=4, heads=8, key_size=48, dense_units=96)
elif model_name == 'CNN2_ATT':
    model = models.CNN2_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                           num_filters=num_filters, dense_units=96, heads=8, key_size=48)

elif model_name == 'CNN2_LSTM':
    model = models.CNN2_LSTM(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                            num_filters=num_filters, lstm_units=48, dense_units=96)

elif model_name == 'CNN2_LSTM_ATT':
    model = models.CNN2_LSTM_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                num_filters=num_filters, lstm_units=48, dense_units=96, heads=8, key_size=48)
elif model_name == 'CNN2_LSTM_TRANS1':
    model = models.CNN2_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=1, heads=8, key_size=48, dense_units=96)
elif model_name == 'CNN2_LSTM_TRANS2':
    model = models.CNN_2LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=2, heads=8, key_size=48, dense_units=96)
elif model_name == 'CNN2_LSTM_TRANS4':
    model = models.CNN2_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=4, heads=8, key_size=48, dense_units=96)
else:
    print("can't find model")

model_name = model_name + '_' + str(pool_size) + '_' + activation + '_' + str(trial)

# compile model model
auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
model.compile(tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=[auroc, aupr])

# fit model
lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_aupr', factor=0.2, patient=4, verbose=1, min_lr=1e-7, mode='max')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_aupr', patience=12, verbose=1, mode='max', restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=200, batch_size=100, validation_data=(x_valid, y_valid), callbacks=[lr_decay, early_stop], verbose=1)

# get positive label sequences and sequence model
pos_index = np.where(y_test[:,0] == 1)[0]   
X = x_test[pos_index]
X_model = model_test[pos_index]

# instantiate explainer class
explainer = explain.Explainer(model, class_index=0)

# calculate attribution maps
saliency_scores = explainer.saliency_maps(X)

# reduce attribution maps to 1D scores
sal_scores = explain.grad_times_input(X, saliency_scores)
saliency_roc, saliency_pr = evaluate.interpretability_performance(sal_scores, X_model, threshold=0.1)
sal_signal, sal_noise_max, sal_noise_mean, sal_noise_topk = evaluate.signal_noise_stats(sal_scores, X_model, top_k=20, threshold=0.1)
snr = evaluate.calculate_snr(sal_signal, sal_noise_topk)

results = model.evaluate(x_test, y_test, verbose=2)
model_auroc = results[1]
stats = np.array([sal_roc, sal_pr, snr, model_auroc])

stats_dir = os.path.join(results_path, model_name+'_stats.npy')
np.save(stats_dir, stats, allow_pickle=True)

# save training and performance results
logs_dir = os.path.join(results_path, model_name+'_logs.pickle')
with open(logs_dir, 'wb') as handle:
    cPickle.dump(history.history, handle)

