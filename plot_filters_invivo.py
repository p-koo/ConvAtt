import numpy as np
import requests as rq
import io, h5py, os
from six.moves import cPickle

import tfomics
import tensorflow as tf
from tfomics import moana, evaluate
import models
import utils
import argparse

#-----------------------------------------------------------------


def load_basset_data(filepath, reverse_compliment=False):
    trainmat = h5py.File(filepath, 'r')
    x_train = np.array(trainmat['train_in']).astype(np.float32)
    y_train = np.array(trainmat['train_out']).astype(np.int32)
    x_valid = np.array(trainmat['valid_in']).astype(np.float32)
    y_valid = np.array(trainmat['valid_out']).astype(np.int32)
    x_test = np.array(trainmat['test_in']).astype(np.float32)
    y_test = np.array(trainmat['test_out']).astype(np.int32)

    x_train = np.squeeze(x_train)
    x_valid = np.squeeze(x_valid)
    x_test = np.squeeze(x_test)
    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])

    if reverse_compliment:
        x_train_rc = x_train[:,::-1,:][:,:,::-1]
        x_valid_rc = x_valid[:,::-1,:][:,:,::-1]
        x_test_rc = x_test[:,::-1,:][:,:,::-1]

        x_train = np.vstack([x_train, x_train_rc])
        x_valid = np.vstack([x_valid, x_valid_rc])
        x_test = np.vstack([x_test, x_test_rc])

        y_train = np.vstack([y_train, y_train])
        y_valid = np.vstack([y_valid, y_valid])
        y_test = np.vstack([y_test, y_test])
        
    return x_train, y_train, x_valid, y_valid, x_test, y_test



def load_deepsea_data(file_path, reverse_compliment=False):
    dataset = h5py.File(file_path, 'r')
    x_train = np.array(dataset['X_train']).astype(np.float32)
    y_train = np.array(dataset['Y_train']).astype(np.float32)
    x_valid = np.array(dataset['X_valid']).astype(np.float32)
    y_valid = np.array(dataset['Y_valid']).astype(np.float32)
    x_test = np.array(dataset['X_test']).astype(np.float32)
    y_test = np.array(dataset['Y_test']).astype(np.float32)

    x_train = np.squeeze(x_train)
    x_valid = np.squeeze(x_valid)
    x_test = np.squeeze(x_test)

    if reverse_compliment:
        x_train_rc = x_train[:,::-1,:][:,:,::-1]
        x_valid_rc = x_valid[:,::-1,:][:,:,::-1]
        x_test_rc = x_test[:,::-1,:][:,:,::-1]
        
        x_train = np.vstack([x_train, x_train_rc])
        x_valid = np.vstack([x_valid, x_valid_rc])
        x_test = np.vstack([x_test, x_test_rc])
        
        y_train = np.vstack([y_train, y_train])
        y_valid = np.vstack([y_valid, y_valid])
        y_test = np.vstack([y_test, y_test])
        
    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])

    return x_train, y_train, x_valid, y_valid, x_test, y_test

#-----------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, help="dataset")
parser.add_argument("-m", type=str, default=0.05, help="model_name")
parser.add_argument("-p", type=int, default=20, help="pool_size")
parser.add_argument("-a", type=str, default='relu', help="activation")
parser.add_argument("-f", type=int, default=64, help="activation")
parser.add_argument("-t", type=int, default=None, help="trial")
args = parser.parse_args()

dataset = args.d
model_name = args.m
pool_size = args.p
activation = args.a
trial = args.t
num_filters = args.f

# set paths
results_path = '../results'
if not os.path.exists(results_path):
    os.makedirs(results_path)
results_path = os.path.join(results_path, dataset)
if not os.path.exists(results_path):
    os.makedirs(results_path)

# load data
data_path = '../../data'
if dataset == 'basset':
    filepath = os.path.join(data_path, 'basset_dataset.h5')
    data = load_basset_data(filepath, reverse_compliment=False)
elif dataset == 'deepsea':
    filepath = os.path.join(data_path, 'deepsea_dataset.h5')
    data = load_deepsea_data(filepath, reverse_compliment=False)
x_train, y_train, x_valid, y_valid, x_test, y_test = data
N, L, A = x_train.shape
num_labels = y_train.shape[1]

# build model
if model_name == 'CNN_ATT':
    model = models.CNN_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                           num_filters=num_filters, dense_units=1024, heads=16, key_size=128)

elif model_name == 'CNN_LSTM':
    model = models.CNN_LSTM(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                            num_filters=num_filters, lstm_units=256, dense_units=1024)

elif model_name == 'CNN_LSTM_ATT':
    model = models.CNN_LSTM_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                num_filters=num_filters, lstm_units=256, dense_units=1024, heads=16, key_size=128)
elif model_name == 'CNN_LSTM_TRANS1':
    model = models.CNN_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=1, heads=16, key_size=256, dense_units=1024)
elif model_name == 'CNN_LSTM_TRANS2':
    model = models.CNN_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=2, heads=16, key_size=256, dense_units=1024)
elif model_name == 'CNN_LSTM_TRANS4':
    model = models.CNN_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                  num_filters=num_filters, num_layers=4, heads=16, key_size=256, dense_units=1024)
else:
    print("can't find model")

model_name = model_name + '_' + str(pool_size) + '_' + activation

# compile model model
auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
model.compile(tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=[auroc, aupr])

# save model params
model_dir = os.path.join(results_path, model_name+'_weights_'+str(trial)+'.h5')
model.load_weights(model_dir)

# Extract ppms from filters
index = [i.name for i in model.layers].index('conv_activation')
ppms = moana.filter_activations(x_test, model, layer=index, window=20,threshold=0.5)

tomtom_dir = os.path.join(results_path, model)
num_filters = moana.count_meme_entries(motif_dir)
stats = tfomics.evaluate.motif_comparison_synthetic_dataset(os.path.join(tomtom_dir,'tomtom.tsv'), num_filters)

# visualize filters
fig = plt.figure(figsize=(25,8))
impress.plot_filters(ppms, fig, num_cols=8, names=stats[2], fontsize=14)
filter_dir = os.path.join(results_path, model_name+'_filters_'+str(trial)+'.pdf')
fig.savefig(filter_dir, format='pdf', dpi=200, bbox_inches='tight')




