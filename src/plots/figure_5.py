# Plots a heatmap with best f1 score from each dataset - architecture

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


model_experiment = {
    'LSTM': ['exp01_lstm', 'exp02_lstm', 'exp03_lstm', 'exp04_lstm', 'exp05_lstm',
             'exp06_lstm'
             ],
    'biLSTM': ['exp01_bilstm', 'exp02_bilstm', 'exp03_bilstm', 'exp04_bilstm', 'exp05_bilstm',
               'exp06_bilstm'
               ],
    'GRU': ['exp01_gru', 'exp02_gru', 'exp03_gru', 'exp04_gru', 'exp05_gru', 'exp06_gru',
            'exp07_gru', 'exp08_gru', 'exp09_gru'
            ],
    'CNN': ['exp01_cnn', 'exp02_cnn', 'exp03_cnn', 'exp04_cnn', 'exp05_cnn',
            'exp06_cnn', 'exp07_cnn', 'exp08_cnn', 'exp09_cnn']
}

datasets = ['activemiles', 'hhar', 'fusion', 'mhealth', 'swell', 'usc-had',
            'uci-har', 'pamap2', 'opportunity', 'realworld']


sns.set_style("whitegrid")
fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(111)

f1_df = pd.DataFrame(index=model_experiment.keys(), columns=datasets, dtype=float)
percent_df = pd.DataFrame(index=model_experiment.keys(), columns=datasets, dtype=float)
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'log')
default_f1 = np.array([0.8] * 5)
for dataset in datasets:
    for model in model_experiment:
        best_avg = 0
        for experiment in model_experiment[model]:
            file_name = '{}_{}.csv'.format(dataset, experiment)
            data_path = os.path.join(log_path, file_name)
            if os.path.isfile(data_path):
                f1_values = np.genfromtxt(data_path, delimiter=',', skip_header=1)[:, 5]
            else:
                f1_values = default_f1
            avg = np.mean(f1_values)
            if avg > best_avg:
                best_avg = avg
                f1_df.loc[model, dataset] = best_avg
        percent_df.loc[:, dataset] = 1 - f1_df.loc[:, dataset] / max(f1_df.loc[:, dataset])

color = 'Blues'
new_cmap = LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=color, a=0.2, b=0.6),
                                             plt.get_cmap(color)(np.linspace(0.2, 0.6, 100)))

pl_cnn = [x * 100 for x in (percent_df.values[3, :]).tolist() if x > 0]
pl_gru = [x * 100 for x in (percent_df.values[2, :]).tolist() if x > 0]
pl_bil = [x * 100 for x in (percent_df.values[1, :]).tolist() if x > 0]
pl_lst = [x * 100 for x in (percent_df.values[0, :]).tolist() if x > 0]
print('Average performance loss 1-CNN, 2-GRU: {}'.format(round(sum(pl_cnn) / len(pl_cnn), 3)))
print('Average performance loss 1-GRU, 2-CNN: {}'.format(round(sum(pl_gru) / len(pl_gru), 3)))
print('Average performance loss biLSTM: {}'.format(round(sum(pl_bil) / len(pl_bil), 3)))
print('Average performance loss LSTM: {}'.format(round(sum(pl_lst) / len(pl_lst), 3)))

nrows = len(f1_df)
ncols = len(f1_df.columns)
for i in range(ncols):
    truthar = [True] * ncols
    truthar[i] = False
    mask = np.array(nrows * [truthar], dtype=bool)
    mask[:, i] = np.array(f1_df.ix[:, i] == 0, dtype=bool)
    red = np.ma.masked_where(mask, f1_df)
    ax.pcolormesh(red, cmap=new_cmap)

box_shift = 0.05
for y in range(f1_df.shape[0]):
    for x in range(f1_df.shape[1]):
        font_weight = 'regular'
        f1_shift = 0.65
        percent_text = '-{:3.2f}%'.format(round(percent_df.ix[y, x] * 100, 2))
        f1_text = '{:2.2f}'.format(round(f1_df.ix[y, x] * 100, 2))
        if f1_df.ix[y, x] == max(f1_df.ix[:, x]):
            ax.add_line(Line2D([x + box_shift, x + box_shift, x + 1 - box_shift, x + 1 - box_shift, x + box_shift],
                               [y + box_shift, y + 1 - box_shift, y + 1 - box_shift, y + box_shift, y + box_shift],
                               color='black', alpha=0.6, linestyle='dashed'))
            f1_shift = 0.5
            percent_text = ''
            font_weight = 'demi'
        ax.text(x + .5, y + f1_shift, f1_text, horizontalalignment='center', verticalalignment='center',
                size=17, weight=font_weight, color='black')
        ax.text(x + .5, y + .25, percent_text, horizontalalignment='center', verticalalignment='center',
                size=17, weight=font_weight)

ax.xaxis.tick_top()
ax.set_xlabel('Dataset', size=15, weight='bold')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Architecture', size=15, weight='bold')
ax.set_xticks(np.arange(ncols) + 0.5)
ax.set_xticklabels(datasets, size=13)
ax.set_yticks(np.arange(nrows) + 0.5)
ax.set_yticklabels(f1_df.index.values, size=13)

ax.set_xticks(np.arange(ncols), minor=True)
ax.set_yticks(np.arange(nrows), minor=True)

# Gridlines based on minor ticks
ax.xaxis.grid(which='minor', linestyle='-', linewidth=1)

ax.spines['bottom'].set_visible(False)

plt.show()

