import os
import tfomics
from tfomics import moana
import numpy as np
import matplotlib.pyplot as plt

results_path = '../results/deepsea'
jaspar_dir = 'motif_database.txt'

for motif_dir in os.listdir('.'):
    if 'filters' in motif_dir:
        split = motif_dir.split('_')
        model_name = 'CNN_'+split[1]+'_'+split[2]+'_'+split[3]
        trial = split[-1][0]
        
        # run tomtom
        tomtom_dir = model_name + '_' + str(trial)
        if not os.path.isdir(tomtom_dir):
            print(model_name, trial)
            output = moana.tomtom(motif_dir, jaspar_dir, tomtom_dir, evalue=False, thresh=0.5, dist='pearson', png=None, tomtom_path='tomtom')

            # perform tomtom analysis
            num_filters = moana.count_meme_entries(motif_dir)
            stats = tfomics.evaluate.motif_comparison_synthetic_dataset(os.path.join(tomtom_dir,'tomtom.tsv'), num_filters)
            stats_dir = os.path.join(results_path, model_name+'_stats_'+str(trial)+'.npy')
            np.save(stats_dir, stats, allow_pickle=True)

            #fig = plt.figure(figsize=(25,8))
            #impress.plot_filters(filters, fig, num_cols=8, names=stats[2], fontsize=14)
