import numpy as np
import requests as rq
import io, h5py, os
from six.moves import cPickle

import tfomics
import tensorflow as tf
from tfomics import moana, evaluate, impress
import models
import utils
import argparse
import matplotlib.pyplot as plt

#-----------------------------------------------------------------

def load_synthetic_data(file_path):

    with h5py.File(io.BytesIO(file_path), 'r') as dataset:
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
parser.add_argument("-f", type=int, default=64, help="filters")
parser.add_argument("-t", type=int, default=None, help="trial")
args = parser.parse_args()

dataset = args.d
model_name = args.m
pool_size = args.p
activation = args.a
trial = args.t
num_filters = args.f

# set paths
results_path = '../results_synthetic'
if not os.path.exists(results_path):
    os.makedirs(results_path)
results_path = os.path.join(results_path, dataset)
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
                           num_filters=num_filters, dense_units=512, heads=8, key_size=128)

elif model_name == 'CNN_LSTM':
    model = models.CNN_LSTM(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                            num_filters=num_filters, lstm_units=128, dense_units=512)

elif model_name == 'CNN_LSTM_ATT':
    model = models.CNN_LSTM_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                num_filters=num_filters, lstm_units=128, dense_units=512, heads=8, key_size=128)
elif model_name == 'CNN_LSTM_TRANS1':
    model = models.CNN_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=1, heads=8, key_size=128, dense_units=512)
elif model_name == 'CNN_LSTM_TRANS2':
    model = models.CNN_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=2, heads=8, key_size=128, dense_units=512)
elif model_name == 'CNN_LSTM_TRANS4':
    model = models.CNN_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=4, heads=8, key_size=128, dense_units=512)
elif model_name == 'CNN2_ATT':
    model = models.CNN2_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                           num_filters=num_filters, dense_units=512, heads=8, key_size=128)

elif model_name == 'CNN2_LSTM':
    model = models.CNN2_LSTM(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                            num_filters=num_filters, lstm_units=128, dense_units=512)

elif model_name == 'CNN2_LSTM_ATT':
    model = models.CNN2_LSTM_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                num_filters=num_filters, lstm_units=128, dense_units=512, heads=8, key_size=128)
elif model_name == 'CNN2_LSTM_TRANS1':
    model = models.CNN2_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=1, heads=8, key_size=128, dense_units=512)
elif model_name == 'CNN2_LSTM_TRANS2':
    model = models.CNN_2LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=2, heads=8, key_size=128, dense_units=512)
elif model_name == 'CNN2_LSTM_TRANS4':
    model = models.CNN2_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=4, heads=8, key_size=128, dense_units=512)
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
history = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_data=(x_valid, y_valid), callbacks=[lr_decay, early_stop], verbose=1)

# save training and performance results
results = model.evaluate(x_test, y_test)
logs_dir = os.path.join(results_path, model_name+'_logs.pickle')
with open(logs_dir, 'wb') as handle:
    cPickle.dump(history.history, handle)
    cPickle.dump(results, handle)

# save model params
model_dir = os.path.join(results_path, model_name+'_weights.h5')
model.save_weights(model_dir)

# Extract ppms from filters
index = [i.name for i in model.layers].index('conv_activation')
ppms = moana.filter_activations(x_test, model, layer=index, window=20, threshold=0.5)

# generate meme file
ppms = moana.clip_filters(ppms, threshold=0.5, pad=3)
motif_dir = os.path.join(results_path, model_name+'_filters.txt')
moana.meme_generate(ppms, output_file=motif_dir, prefix='filter')

# Tomtom analysis
tomtom_dir = os.path.join(results_path, model_name)
jaspar_dir = 'motif_database.txt'
output = moana.tomtom(motif_dir, jaspar_dir, tomtom_dir, evalue=False, thresh=0.5, dist='pearson', png=None, tomtom_path='tomtom')

# motif analysis
num_filters = moana.count_meme_entries(motif_dir)
stats = tfomics.evaluate.motif_comparison_synthetic_dataset(os.path.join(tomtom_dir,'tomtom.tsv'), num_filters)
stats_dir = os.path.join(results_path, model_name+'_stats.npy')
np.save(stats_dir, stats, allow_pickle=True)

# visualize filters
fig = plt.figure(figsize=(25,8))
impress.plot_filters(ppms, fig, num_cols=8, names=stats[2], fontsize=14)
filter_dir = os.path.join(results_path, model_name+'_filters.pdf')
fig.savefig(filter_dir, format='pdf', dpi=200, bbox_inches='tight')



