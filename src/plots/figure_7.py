# Plots a heatmap with fastest average training dataset - architecture

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

sns.set_style("whitegrid")
fig = plt.figure(figsize=(15, 3.4))
ax = fig.add_subplot(111)

model_experiment = {
    'LSTM': {'exp01_lstm': 128400, 'exp02_lstm': 496800, 'exp03_lstm': 771000,
             'exp04_lstm': 1105200, 'exp05_lstm': 1499400, 'exp06_lstm': 1953600},
    'biLSTM': {'exp01_bilstm': 256800, 'exp02_bilstm': 993600, 'exp03_bilstm': 1542000,
               'exp04_bilstm': 2210400, 'exp05_bilstm': 2998800, 'exp06_bilstm': 3907200},
    'GRU': {'exp01_gru': 96300, 'exp02_gru': 372600, 'exp03_gru': 578250,
            'exp04_gru': 828900, 'exp05_gru': 1124550, 'exp06_gru': 1465200,
            'exp07_gru': 1850850, 'exp08_gru': 2281500, 'exp09_gru': 2757150},
    'CNN': {'exp01_cnn': 1144676, 'exp02_cnn': 1176676, 'exp03_cnn': 1274212,
            'exp04_cnn': 1306212, 'exp05_cnn': 748388, 'exp06_cnn': 845924,
            'exp07_cnn': 943460, 'exp08_cnn': 673380, 'exp09_cnn': 742244}
}

datasets = {'activemiles': 32548, 'hhar': 151840, 'fusion': 39523, 'mhealth': 8566,
            'swell': 8064, 'usc-had': 17412, 'uci-har': 8771, 'pamap2': 143572,
            'opportunity': 36169, 'realworld': 125411}

plot_df = pd.DataFrame(index=model_experiment.keys(), columns=datasets, dtype=float)
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'log')

for dataset in datasets:
    for model in model_experiment:
        exp_results = np.zeros((len(model_experiment[model]), 2), dtype=float)
        for row, experiment in enumerate(model_experiment[model]):
            file_name = '{}_{}.csv'.format(dataset, experiment)
            data_path = os.path.join(log_path, file_name)
            if os.path.isfile(data_path):
                exp_values = np.genfromtxt(data_path, delimiter=',', skip_header=1)
                exp_results[row, 0] = np.mean(exp_values[:, 5])
                exp_results[row, 1] = model_experiment[model][experiment]

        best = np.mean(exp_results[np.flip(exp_results[:, 0].argsort(), axis=0)][0, 1])
        plot_df.loc[model, dataset] = best if best > 0 else 666
    plot_df.loc[:, dataset] = plot_df.loc[:, dataset] / min(plot_df.loc[:, dataset])


a = sum((plot_df.values[2:, ][0]).tolist()) * 10
print('Average speed loss 1-CNN, 2-GRU: {:5.3f}'.format(sum((plot_df.values[2:, ][0]).tolist()) * 10))
print('Average speed loss 1-CNN, 3-biLSTM: {}'.format(sum((plot_df.values[1:, ][0]).tolist()) * 10))
print('Average speed loss 1-CNN, 4-LSTM: {}'.format(sum((plot_df.values[0:, ][0]).tolist()) * 10))

color = 'Blues'
new_cmap = LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=color, a=0.2, b=0.6),
                                             plt.get_cmap(color)(np.linspace(0.2, 0.6, 100)))
nrows = len(plot_df)
ncols = len(plot_df.columns)
for i in range(ncols):
    truthar = [True] * ncols
    truthar[i] = False
    mask = np.array(nrows * [truthar], dtype=bool)
    mask[:, i] = np.array(plot_df.ix[:, i] == 0, dtype=bool)
    red = np.ma.masked_where(mask, -plot_df)
    ax.pcolormesh(red, cmap=new_cmap)

box_shift = 0.05
for y in range(plot_df.shape[0]):
    for x in range(plot_df.shape[1]):
        value = plot_df.ix[y, x]
        font_weight = 'regular'
        text = '{}%'.format(round(value * 100, 2))
        if value == min(plot_df.ix[:, x]):
            ax.add_line(Line2D([x + box_shift, x + box_shift, x + 1 - box_shift, x + 1 - box_shift, x + box_shift],
                               [y + box_shift, y + 1 - box_shift, y + 1 - box_shift, y + box_shift, y + box_shift],
                               color='black', alpha=0.6, linestyle='dashed'))
            font_weight = 'demi'
        ax.text(x + .5, y + .5, text, horizontalalignment='center', verticalalignment='center',
                size=15, weight=font_weight)

ax.xaxis.tick_top()
ax.set_xlabel('Dataset', size=14, weight='bold')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Architecture', size=14, weight='bold')
ax.set_xticks(np.arange(ncols) + 0.5)
ax.set_xticklabels(datasets, size=12)
ax.set_yticks(np.arange(nrows) + 0.5)
ax.set_yticklabels(plot_df.index.values, size=12)

ax.set_xticks(np.arange(ncols), minor=True)
ax.set_yticks(np.arange(nrows), minor=True)

# Gridlines based on minor ticks
ax.xaxis.grid(which='minor', linestyle='-', linewidth=1)

ax.spines['bottom'].set_visible(False)

plt.show()

