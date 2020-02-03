# Plots a heatmap with best f1 score from each dataset - architecture

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

model_experiment = {
    'DBN': ['exp01_dbn', 'exp02_dbn', 'exp03_dbn', 'exp04_dbn', 'exp05_dbn',
            'exp06_dbn'
            ],
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
fig, axs = plt.subplots(2, 1, figsize=(15, 8))

ax = axs[0]

f1_df = pd.DataFrame(index=model_experiment.keys(), columns=datasets, dtype=float)
acc_df = pd.DataFrame(index=model_experiment.keys(), columns=datasets, dtype=float)
percent_f1_df = pd.DataFrame(index=model_experiment.keys(), columns=datasets, dtype=float)
percent_acc_df = pd.DataFrame(index=model_experiment.keys(), columns=datasets, dtype=float)
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'log')
default_f1 = np.array([0.8] * 5)
default_acc = np.array([0.8] * 5)
for dataset in datasets:
    for model in model_experiment:
        best_f1_avg = 0
        best_acc_avg = 0
        for experiment in model_experiment[model]:
            file_name = f'{dataset}_{experiment}.csv'
            data_path = os.path.join(log_path, file_name)
            if os.path.isfile(data_path):
                f1_values = np.genfromtxt(data_path, delimiter=',', skip_header=1)[:, 5]
                acc_values = np.genfromtxt(data_path, delimiter=',', skip_header=1)[:, 3]
            else:
                f1_values = default_f1
                acc_values = default_acc
            avg_f1 = np.mean(f1_values)
            avg_acc = np.mean(acc_values)
            if avg_f1 > best_f1_avg:
                best_f1_avg = avg_f1
                f1_df.loc[model, dataset] = best_f1_avg
            if avg_acc > best_acc_avg:
                best_acc_avg = avg_acc
                acc_df.loc[model, dataset] = best_acc_avg
        percent_f1_df.loc[:, dataset] = 1 - f1_df.loc[:, dataset] / max(f1_df.loc[:, dataset])
        percent_acc_df.loc[:, dataset] = 1 - acc_df.loc[:, dataset] / max(acc_df.loc[:, dataset])

color = 'Blues'
new_cmap = LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=color, a=0.2, b=0.6),
                                             plt.get_cmap(color)(np.linspace(0.2, 0.6, 100)))

pl_dbn_f1 = [x * 100 for x in (percent_f1_df.values[0, :]).tolist() if x > 0]
pl_lst_f1 = [x * 100 for x in (percent_f1_df.values[1, :]).tolist() if x > 0]
pl_bil_f1 = [x * 100 for x in (percent_f1_df.values[2, :]).tolist() if x > 0]
pl_gru_f1 = [x * 100 for x in (percent_f1_df.values[3, :]).tolist() if x > 0]
pl_cnn_f1 = [x * 100 for x in (percent_f1_df.values[4, :]).tolist() if x > 0]
print('f1 average performance loss 1-CNN, 2-GRU: {}'.format(round(sum(pl_cnn_f1) / len(pl_cnn_f1), 3)))
print('f1 average performance loss 1-GRU, 2-CNN: {}'.format(round(sum(pl_gru_f1) / len(pl_gru_f1), 3)))
print('f1 average performance loss biLSTM: {}'.format(round(sum(pl_bil_f1) / len(pl_bil_f1), 3)))
print('f1 average performance loss LSTM: {}'.format(round(sum(pl_lst_f1) / len(pl_lst_f1), 3)))
print('f1 average performance loss DBN: {}'.format(round(sum(pl_dbn_f1) / len(pl_dbn_f1), 3)))

pl_dbn_acc = [x * 100 for x in (percent_acc_df.values[0, :]).tolist() if x > 0]
pl_lst_acc = [x * 100 for x in (percent_acc_df.values[1, :]).tolist() if x > 0]
pl_bil_acc = [x * 100 for x in (percent_acc_df.values[2, :]).tolist() if x > 0]
pl_gru_acc = [x * 100 for x in (percent_acc_df.values[3, :]).tolist() if x > 0]
pl_cnn_acc = [x * 100 for x in (percent_acc_df.values[4, :]).tolist() if x > 0]
print('acc average performance loss 1-CNN, 2-GRU: {}'.format(round(sum(pl_cnn_acc) / len(pl_cnn_acc), 3)))
print('acc average performance loss 1-GRU, 2-CNN: {}'.format(round(sum(pl_gru_acc) / len(pl_gru_acc), 3)))
print('acc average performance loss biLSTM: {}'.format(round(sum(pl_bil_acc) / len(pl_bil_acc), 3)))
print('acc average performance loss LSTM: {}'.format(round(sum(pl_lst_acc) / len(pl_lst_acc), 3)))
print('acc average performance loss DBN: {}'.format(round(sum(pl_dbn_acc) / len(pl_dbn_acc), 3)))

nrows = len(f1_df)
ncols = len(f1_df.columns)
for i in range(ncols):
    truthar = [True] * ncols
    truthar[i] = False
    mask = np.array(nrows * [truthar], dtype=bool)
    mask[:, i] = np.array(f1_df.iloc[:, i] == 0, dtype=bool)
    red = np.ma.masked_where(mask, f1_df)
    ax.pcolormesh(red, cmap=new_cmap)

box_shift = 0.05

for y in range(f1_df.shape[0]):
    for x in range(f1_df.shape[1]):
        font_weight = 'regular'
        f1_shift = 0.65
        percent_text = '-{:3.2f}%'.format(round(percent_f1_df.iloc[y, x] * 100, 2))
        f1_text = '{:2.2f}'.format(round(f1_df.iloc[y, x] * 100, 2))
        if f1_df.iloc[y, x] == max(f1_df.iloc[:, x]):
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
ax.set_xlabel('(a) f1', size=15, weight='bold')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Architecture', size=15, weight='bold')
ax.set_xticks(np.arange(ncols) + 0.5)
ax.set_xticklabels(datasets, size=13, weight='bold')
ax.set_yticks(np.arange(nrows) + 0.5)
ax.set_yticklabels(f1_df.index.values, size=13)

ax.set_xticks(np.arange(ncols), minor=True)
ax.set_yticks(np.arange(nrows), minor=True)

# Gridlines based on minor ticks
ax.xaxis.grid(which='minor', linestyle='-', linewidth=1)

ax.spines['bottom'].set_visible(False)


ax = axs[1]
for i in range(ncols):
    truthar = [True] * ncols
    truthar[i] = False
    mask = np.array(nrows * [truthar], dtype=bool)
    mask[:, i] = np.array(f1_df.iloc[:, i] == 0, dtype=bool)
    red = np.ma.masked_where(mask, f1_df)
    ax.pcolormesh(red, cmap=new_cmap)

for y in range(acc_df.shape[0]):
    for x in range(acc_df.shape[1]):
        font_weight = 'regular'
        acc_shift = 0.65
        percent_text = '-{:3.2f}%'.format(round(percent_acc_df.iloc[y, x] * 100, 2))
        acc_text = '{:2.2f}'.format(round(acc_df.iloc[y, x] * 100, 2))
        if acc_df.iloc[y, x] == max(acc_df.iloc[:, x]):
            ax.add_line(Line2D([x + box_shift, x + box_shift, x + 1 - box_shift, x + 1 - box_shift, x + box_shift],
                               [y + box_shift, y + 1 - box_shift, y + 1 - box_shift, y + box_shift, y + box_shift],
                               color='black', alpha=0.6, linestyle='dashed'))
            acc_shift = 0.5
            percent_text = ''
            font_weight = 'demi'
        ax.text(x + .5, y + acc_shift, acc_text, horizontalalignment='center', verticalalignment='center',
                size=17, weight=font_weight, color='black')
        ax.text(x + .5, y + .25, percent_text, horizontalalignment='center', verticalalignment='center',
                size=17, weight=font_weight)

ax.xaxis.tick_top()
ax.set_xlabel('(b) accuracy', size=15, weight='bold')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Architecture', size=15, weight='bold')
ax.set_xticks(np.arange(ncols) + 0.5)
ax.set_xticklabels(datasets, size=13, weight='bold')
ax.set_yticks(np.arange(nrows) + 0.5)
ax.set_yticklabels(acc_df.index.values, size=13)

ax.set_xticks(np.arange(ncols), minor=True)
ax.set_yticks(np.arange(nrows), minor=True)

# Gridlines based on minor ticks
ax.xaxis.grid(which='minor', linestyle='-', linewidth=1)

ax.spines['bottom'].set_visible(False)

plt.show()
