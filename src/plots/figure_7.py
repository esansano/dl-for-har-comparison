import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


model_experiment = {
    'LSTM': [['exp01_lstm', 'exp02_lstm', 'exp03_lstm', 'exp04_lstm', 'exp05_lstm',
             'exp06_lstm', 'exp07_lstm', 'exp08_lstm', 'exp09_lstm', 'exp10_lstm'],
             ['200', '250', '300', '350', '400', '450', '500', '550', '600'], 'hidden units',
             'average f1 improvement (%)', 'average time increase (%)', (0, 0)
             ],
    'biLSTM': [['exp01_bilstm', 'exp02_bilstm', 'exp03_bilstm', 'exp04_bilstm', 'exp05_bilstm',
               'exp06_bilstm', 'exp07_bilstm', 'exp08_bilstm'],
               ['200', '250', '300', '350', '400', '450', '500'], 'hidden units',
               'average f1 improvement (%)', 'average time increase (%)', (0, 1)
               ],
    'GRU': [['exp01_gru', 'exp02_gru', 'exp03_gru', 'exp04_gru', 'exp05_gru', 'exp06_gru',
            'exp07_gru', 'exp08_gru', 'exp09_gru', 'exp10_gru'],
            ['200', '250', '300', '350', '400', '450', '500', '550', '600'], 'hidden units',
            'average f1 improvement (%)', 'average time increase (%)', (0, 2)
            ],
    'CNN': [['exp01_cnn', 'exp02_cnn', 'exp03_cnn', 'exp04_cnn', 'exp05_cnn',
            'exp06_cnn', 'exp07_cnn', 'exp08_cnn', 'exp09_cnn'],
            ['17', '15', '13', '11', '9', '7', '5', '3'], 'filter temporal width',
            'average f1 improvement (%)', 'average time increase (%)', (1, 0)
            ],
    'DBN': [['exp01_dbn', 'exp02_dbn', 'exp03_dbn', 'exp04_dbn', 'exp05_dbn',
            'exp06_dbn'],
            ['2', '3', '4', '5', '6'], 'number of layers',
            'average f1 improvement (%)', 'average time increase (%)', (1, 1)
            ]
}

datasets = {
    'activemiles': 32548,
    'hhar': 151840,
    'fusion': 39523,
    'mhealth': 8566,
    'swell': 8064,
    'usc-had': 17412,
    'uci-har': 8771,
    'pamap2': 143572,
    'opportunity': 36169,
    'realworld': 125411
}
sns.set_style("whitegrid")
fig, axs = plt.subplots(2, 3, figsize=(14, 8))

for model in model_experiment:
    plt_ax = model_experiment[model][-1]
    ax = axs[plt_ax]

    f1_df = pd.DataFrame(index=model_experiment[model][0], columns=datasets, dtype=float)
    time_df = pd.DataFrame(index=model_experiment[model][0], columns=datasets, dtype=float)
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'log')

    for experiment in model_experiment[model][0]:
        for dataset in datasets:
            file_name = '{}_{}.csv'.format(dataset, experiment)
            data_path = os.path.join(log_path, file_name)
            exp_avg = 0
            if os.path.isfile(data_path):
                exp_values = np.genfromtxt(data_path, delimiter=',', skip_header=1)
                exp_avg = np.mean(exp_values[:, 5])
                exp_time = exp_values[:, 1]
                exp_epochs = exp_values[:, 2] + 50
                time_per_epoch = exp_time / exp_epochs
                time_dp_avg = np.mean(time_per_epoch / datasets[dataset]) * 10000
            else:
                exp_avg = 1
                time_dp_avg = 0
            f1_df.loc[experiment, dataset] = exp_avg
            time_df.loc[experiment, dataset] = time_dp_avg

    f1_df.loc[:] = (f1_df.loc[:] / f1_df.loc[model_experiment[model][0][0], :]) * 100 - 100
    time_df.loc[:] = (time_df.loc[:] / time_df.loc[model_experiment[model][0][0], :]) * 100 - 100

    values_f1 = f1_df.values
    values_time = time_df.values

    color = 'Blues'
    bl_cmap = LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=color, a=0.2, b=0.6),
                                                plt.get_cmap(color)(np.linspace(0.2, 0.6, 100)))

    color = 'Reds'
    rd_cmap = LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=color, a=0.2, b=0.6),
                                                plt.get_cmap(color)(np.linspace(0.2, 0.6, 100)))

    color_f1 = bl_cmap(0.95)
    color_tm = rd_cmap(0.35)
    # plt.tick_params(axis='both', which='major', labelsize=14)

    x = np.arange(values_f1.shape[0])
    est = np.mean(values_f1, axis=1)
    sd = np.std(values_f1, axis=1)
    cis = (est - sd, est + sd)
    cis_size = np.max(cis) - np.min(cis)
    ax.fill_between(x[1:] - 1, cis[0][1:], cis[1][1:], color=color_f1, alpha=0.15)
    ax.plot(est[1:], linewidth=4, color=color_f1, alpha=1)

    ax.set_xticklabels(model_experiment[model][1])
    # ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(direction='out', length=8, width=3, color='darkgray')
    ax.set_xlabel(model_experiment[model][2], size=15, weight='regular')

    ax.set_ylabel('')
    # ax.set_ylabel(model_experiment[model][3], size=16, weight='regular')
    ax.grid(which='both', axis='y', linestyle='')
    ax.grid(which='major', axis='x', linestyle='')
    # ax.axes.set(xlim=(-0.2, 4.1))
    ax.set_xticks(np.arange(0, len(model_experiment[model][1]), 1))

    max_f1 = np.max(cis)
    min_f1 = np.min(cis)

    max_time = np.max(values_time)
    min_time = np.min(values_time)
    v = (max_time - min_time) / (max_f1 - min_f1)
    v1 = max_time / v
    v2 = min_time / v
    values_time = values_time / v

    x = np.arange(values_time.shape[0])
    est = np.mean(values_time, axis=1)
    print(model, len(est))
    ax.plot(est[1:], linewidth=4, color=color_tm, alpha=1, linestyle='--', marker='o', markersize=10)
    for i, value in enumerate(est[1:]):
        text = str(int(value * v)) + '%'
        ax.annotate(text, (i - 0.2, est[i + 1] + cis_size / 20), color=color_tm, size=12, weight='bold')
        # print(est)

    ax.grid(which='both', axis='y', linestyle='')
    ax.grid(which='major', axis='x', linestyle='')
    ax.set_frame_on(False)
    ax.set_title(model, y=0.85)

ax = axs[1, 2]
ax.grid(which='both', axis='y', linestyle='')
ax.grid(which='major', axis='x', linestyle='')
ax.set_frame_on(False)
custom_lines = [Line2D([0], [0], color=color_f1, lw=5), Line2D([0], [0], color=color_tm, lw=5, linestyle='--')]
ax.legend(custom_lines, ['average f1 improvement (%)', 'average time increase (%)'], fontsize='x-large', shadow=True,
          bbox_to_anchor=(1.00, 0.75))
ax.legend()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
fig.tight_layout()

plt.show()
