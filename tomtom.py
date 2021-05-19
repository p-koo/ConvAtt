import os
import numpy as np
from six.moves import cPickle

import tfomics
from tfomics import moana, evaluate, impress


base_models = ['CNN', 'CNN_ATT', 'CNN_LSTM_ATT', 'CNN_LSTM_TRANS1', 'CNN_LSTM_TRANS2', 'CNN_LSTM_TRANS4',
               'CNN2', 'CNN2_ATT', 'CNN2_LSTM_ATT', 'CNN2_LSTM_TRANS1', 'CNN2_LSTM_TRANS2', 'CNN2_LSTM_TRANS4',
               'CNN_LSTM2', 'CNN_LSTM2_ATT', 'CNN_LSTM2_TRANS1', 'CNN_LSTM2_TRANS2', 'CNN_LSTM2_TRANS4',]
activations = ['relu', 'exponential']
num_trials = 5


results_path = '../results_task1'
save_path = '../results_tomtom'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for model in base_models:
    for activation in activations:
        trial_stats = []
        for trial in range(num_trials):
            model_name = model + '_' + activation + '_' + str(trial)

			# Tomtom analysis
			motif_dir = os.path.join(results_path, model_name+'_filters.txt')
			tomtom_dir = os.path.join(save_path, model_name)
			jaspar_dir = 'motif_database.txt'
			output = moana.tomtom(motif_dir, jaspar_dir, tomtom_dir, evalue=False, thresh=0.1, dist='pearson', png=None, tomtom_path='tomtom')

			# motif analysis
			num_filters = moana.count_meme_entries(motif_dir)
			stats = tfomics.evaluate.motif_comparison_synthetic_dataset(os.path.join(tomtom_dir,'tomtom.tsv'), num_filters)
			stats_dir = os.path.join(save_path, model_name+'_stats.npy')
			np.save(stats_dir, stats, allow_pickle=True)
