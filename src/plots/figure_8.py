import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

experiments = ['exp06_cnn', 'exp09_lstm', 'exp09_gru', 'exp07_bilstm']
datasets = ['fusion', 'mhealth', 'uci_har']
models = {'CNN': [], 'LSTM': [], 'GRU': [], 'BILSTM': []}
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'log', 'loo')

for model, experiment in zip(models, experiments):
    for dataset in datasets:
        name = f'{dataset}_{experiment}.csv'
        file = os.path.join(log_path, name)
        if os.path.isfile(file):
            data = pd.read_csv(file).to_numpy()
            f1_test = np.mean(data[:, 5])
            f1_val = np.mean(data[:, 3])
            print(f'{dataset:15s}{experiment:15s}{f1_val:10.4f}{f1_test:10.4f}')
            models[model].append(f1_test)

sns.set(context='paper', style='ticks', rc={'lines.linewidth': 0.7})
fig = plt.figure(dpi=200)
fig.set_size_inches(8, 4)
sns.set(font_scale=0.90)
plt.margins(x=0, y=0)
ax = fig.add_subplot(111)
ax.set_facecolor('white')

barWidth = 0.20

bars_lstm = models['LSTM']
bars_bilstm = models['BILSTM']
bars_gru = models['GRU']
bars_cnn = models['CNN']

r1 = np.arange(len(bars_lstm))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, bars_lstm, color='#BAD6EA', width=barWidth, edgecolor='white', label='LSTM')
plt.bar(r2, bars_bilstm, color='#88BEDC', width=barWidth, edgecolor='white', label='BILSTM')
plt.bar(r3, bars_gru, color='#539DCC', width=barWidth, edgecolor='white', label='GRU')
plt.bar(r4, bars_cnn, color='#2A7AB9', width=barWidth, edgecolor='white', label='CNN')

plt.xlabel('Dataset', fontweight='bold')
plt.ylabel('f1', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars_lstm))], datasets)

plt.legend()
plt.show()
