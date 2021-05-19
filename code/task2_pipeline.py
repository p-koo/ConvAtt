import numpy as np
import h5py, os
from six.moves import cPickle
import argparse
import matplotlib.pyplot as plt

import tensorflow as tf
import tfomics
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
        model_test = np.array(dataset['model_test']).astype(np.float32).transpose([0, 2, 1])

    return x_train, y_train, x_valid, y_valid, x_test, y_test, model_test

#-----------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-m", type=str, default=0.05, help="model_name")
parser.add_argument("-a", type=str, default='relu', help="activation")
parser.add_argument("-f", type=int, default=48, help="filters")
parser.add_argument("-t", type=int, default=None, help="trial")
args = parser.parse_args()

model_name = args.m
activation = args.a
trial = args.t
num_filters = args.f

# set paths
results_path = '../results_task2'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# load data
data_path = '../../data'
filepath = os.path.join(data_path, 'synthetic_dataset.h5')
data = load_synthetic_data(filepath)
x_train, y_train, x_valid, y_valid, x_test, y_test, model_test = data
N, L, A = x_train.shape
num_labels = y_train.shape[1]

# build model
if model_name == 'CNN':
    model = models.CNN(in_shape=(L,A), num_out=num_labels, activation=activation, 
                           num_filters=num_filters, dense_units=96, heads=8, key_size=128)
elif model_name == 'CNN2':
    model = models.CNN2(in_shape=(L,A), num_out=num_labels, activation=activation, 
                           num_filters=num_filters, dense_units=96, heads=8, key_size=128)
elif model_name == 'CNN_ATT':
    model = models.CNN_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, 
                           num_filters=num_filters, dense_units=96, heads=8, key_size=48)

elif model_name == 'CNN_LSTM':
    model = models.CNN_LSTM(in_shape=(L,A), num_out=num_labels, activation=activation, 
                            num_filters=num_filters, lstm_units=48, dense_units=96)

elif model_name == 'CNN_LSTM_ATT':
    model = models.CNN_LSTM_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, 
                                num_filters=num_filters, lstm_units=48, dense_units=96, heads=8, key_size=96)
elif model_name == 'CNN_LSTM_TRANS1':
    model = models.CNN_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, 
                                  num_filters=num_filters, num_layers=1, heads=8, key_size=48, dense_units=96)
elif model_name == 'CNN_LSTM_TRANS2':
    model = models.CNN_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, 
                                  num_filters=num_filters, num_layers=2, heads=8, key_size=48, dense_units=96)
elif model_name == 'CNN_LSTM_TRANS4':
    model = models.CNN_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, 
                                  num_filters=num_filters, num_layers=4, heads=8, key_size=48, dense_units=96)

elif model_name == 'CNN_LSTM2':
    model = models.CNN_LSTM2(in_shape=(L,A), num_out=num_labels, activation=activation, 
                            num_filters=num_filters, lstm_units=48, dense_units=96)

elif model_name == 'CNN_LSTM2_ATT':
    model = models.CNN_LSTM2_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, 
                                num_filters=num_filters, lstm_units=48, dense_units=96, heads=8, key_size=96)
elif model_name == 'CNN_LSTM2_TRANS1':
    model = models.CNN_LSTM2_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, 
                                  num_filters=num_filters, num_layers=1, heads=8, key_size=48, dense_units=96)
elif model_name == 'CNN_LSTM2_TRANS2':
    model = models.CNN_LSTM2_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, 
                                  num_filters=num_filters, num_layers=2, heads=8, key_size=48, dense_units=96)
elif model_name == 'CNN_LSTM2_TRANS4':
    model = models.CNN_LSTM2_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, 
                                  num_filters=num_filters, num_layers=4, heads=8, key_size=48, dense_units=96)

elif model_name == 'CNN2_ATT':
    model = models.CNN2_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, 
                           num_filters=num_filters, dense_units=96, heads=8, key_size=48)

elif model_name == 'CNN2_LSTM':
    model = models.CNN2_LSTM(in_shape=(L,A), num_out=num_labels, activation=activation, 
                            num_filters=num_filters, lstm_units=48, dense_units=96)

elif model_name == 'CNN2_LSTM_ATT':
    model = models.CNN2_LSTM_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, 
                                num_filters=num_filters, lstm_units=48, dense_units=96, heads=8, key_size=48)
elif model_name == 'CNN2_LSTM_TRANS1':
    model = models.CNN2_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, 
                                  num_filters=num_filters, num_layers=1, heads=8, key_size=48, dense_units=96)
elif model_name == 'CNN2_LSTM_TRANS2':
    model = models.CNN2_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, 
                                  num_filters=num_filters, num_layers=2, heads=8, key_size=48, dense_units=96)
elif model_name == 'CNN2_LSTM_TRANS4':
    model = models.CNN2_LSTM_TRANS(in_shape=(L,A), num_out=num_labels, activation=activation, 
                                  num_filters=num_filters, num_layers=4, heads=8, key_size=48, dense_units=96)
else:
    print("can't find model")

model_name = model_name + '_' + activation + '_' + str(trial)



# set up optimizer and metrics
auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
model.compile(optimizer=optimizer, loss=loss, metrics=[auroc, aupr])


# early stopping callback
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_auroc', 
                                            patience=10, 
                                            verbose=1, 
                                            mode='max', 
                                            restore_best_weights=True)
# reduce learning rate callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auroc', 
                                                factor=0.2,
                                                patience=4, 
                                                min_lr=1e-7,
                                                mode='max',
                                                verbose=1) 

# train model
history = model.fit(x_train, y_train, 
                    epochs=100,
                    batch_size=100, 
                    shuffle=True,
                    validation_data=(x_valid, y_valid), 
                    callbacks=[es_callback, reduce_lr])

# test model 
results = model.evaluate(x_test, y_test, verbose=2)
model_auroc = results[1]

# get positive label sequences and sequence model
pos_index = np.where(y_test[:,0] == 1)[0]   
X = x_test[pos_index]
X_model = model_test[pos_index]

# instantiate explainer class
explainer = tfomics.explain.Explainer(model, class_index=0)

# calculate attribution maps
saliency_scores = explainer.saliency_maps(X)

# reduce attribution maps to 1D scores
sal_scores = tfomics.explain.grad_times_input(X, saliency_scores)

# compare distribution of attribution scores at positions with and without motifs
saliency_roc, saliency_pr = tfomics.evaluate.interpretability_performance(sal_scores, X_model, threshold=0.1)
sal_signal, sal_noise_max, sal_noise_mean, sal_noise_topk = tfomics.evaluate.signal_noise_stats(sal_scores, X_model, top_k=20, threshold=0.1)
snr = tfomics.evaluate.calculate_snr(sal_signal, sal_noise_topk)

print(model_name)
print(model_auroc)
print("%s: %.3f+/-%.3f"%('saliency', np.mean(saliency_roc), np.std(saliency_roc)))
print("%s: %.3f+/-%.3f"%('saliency', np.mean(saliency_pr), np.std(saliency_pr)))

# save results
stats_dir = os.path.join(results_path, model_name+'_stats.pickle')
with open(stats_dir, 'wb') as handle:
    cPickle.dump(model_auroc, handle)
    cPickle.dump(np.mean(saliency_roc), handle)
    cPickle.dump(np.mean(saliency_pr), handle)
    cPickle.dump(np.mean(snr), handle)

