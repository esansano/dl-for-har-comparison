import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

experiments = ['exp06_cnn', 'exp09_lstm', 'exp09_gru', 'exp07_bilstm']
datasets = ['fusion', 'mhealth', 'uci_har']
model_avg = {'CNN': [], 'LSTM': [], 'GRU': [], 'BILSTM': []}
model_sd = {'CNN': [], 'LSTM': [], 'GRU': [], 'BILSTM': []}

log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'log', 'loo')

for model, experiment in zip(model_avg, experiments):
    for dataset in datasets:
        name = f'{dataset}_{experiment}.csv'
        file = os.path.join(log_path, name)
        if os.path.isfile(file):
            data = pd.read_csv(file).to_numpy()
            f1_test = np.mean(data[:, 5])
            f1_val = np.mean(data[:, 3])
            sd_test = np.std(data[:, 5])
            print(f'{dataset:15s}{experiment:15s}{f1_val:10.4f}{f1_test:10.4f}')
            model_avg[model].append(f1_test)
            model_sd[model].append(sd_test)

sns.set(context='paper', style='ticks', rc={'lines.linewidth': 0.7})
fig = plt.figure(dpi=200)
fig.set_size_inches(8, 4)
sns.set(font_scale=0.90)
plt.margins(x=0, y=0)
ax = fig.add_subplot(111)
ax.set_facecolor('white')

barWidth = 0.20

bars_lstm = model_avg['LSTM']
bars_bilstm = model_avg['BILSTM']
bars_gru = model_avg['GRU']
bars_cnn = model_avg['CNN']

r1 = np.arange(len(bars_lstm))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, bars_lstm, yerr=model_sd['LSTM'], capsize=5, ecolor='#444444', color='#BAD6EA', width=barWidth,
        edgecolor='white', label='LSTM')
plt.bar(r2, bars_bilstm, yerr=model_sd['BILSTM'], capsize=5, ecolor='#444444', color='#88BEDC', width=barWidth,
        edgecolor='white', label='BILSTM')
plt.bar(r3, bars_gru, yerr=model_sd['GRU'], capsize=5, ecolor='#444444', color='#539DCC', width=barWidth,
        edgecolor='white', label='GRU')
plt.bar(r4, bars_cnn, yerr=model_sd['CNN'], capsize=5, ecolor='#444444', color='#2A7AB9', width=barWidth,
        edgecolor='white', label='CNN')

plt.xlabel('Dataset', fontweight='bold')
plt.ylabel('f1', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars_lstm))], datasets)

plt.legend()
plt.show()
