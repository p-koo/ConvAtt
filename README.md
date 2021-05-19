# Interpretability Analysis for Convolutional-Hybrid Architectures

This repository explores different convolutional-hybrid architectures to quantitatively analyze the extent that first layer filters learn motif representations (Task 1) and quantitatively measure how reliable their saliency maps recapitulate ground truth patterns (Task 2).

#### Dependencies
- Python 3.5 or greater
- Pandas, NumPy, Matplotlib, H5py
- TFomics -- https://github.com/p-koo/tfomics 
- TensorFlow 2.0 or greater
- Logomaker (Tareen and Kinney)

#### Source files
- models.py - many hybrid model architectures 
- task1_pipeline.py - callable function to fit synthetic multi-task data and perform tomtom analysis for first layer filters
- task2_pipeline.py - callable function to fit sythetic regulatory code data and perform saliency analysis and quantify efficacy of saliency maps
- commands_task1.py - list of function calls to task1_pipeline.py for various hybrid architectures
- commands_task2.py - list of function calls to task2_pipeline.py for various hybrid architectures
- Analyze_task1_results.ipynb - notebook to reproduce figure 1
- Analyze_task2_results.ipynb - notebook to reproduce figure 2

#### Data
- synthetic_dataset.h5 - data used in Koo & Eddy, PLoS Comp Bio (2019) -- multi-task classification for 12 TF binding sites
- synthetic_code_dataset.h5 - data used in Koo & Ploenzke, Nature Machine Intelligence (2021) -- synthetic regulatory code prediction task
