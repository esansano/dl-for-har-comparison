# Plots a violinplot of the distribution of acc and gyr values

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import datamanager as dm

sns.set(context='paper', style='ticks', rc={'lines.linewidth': 0.7})

fig = plt.figure(dpi=200)
fig.set_size_inches(8, 4.5)
ax = fig.add_subplot(111)
sns.set(font_scale=0.50)
plt.margins(x=0, y=0)

min_size = 3500000
datasets = ['activemiles', 'hhar', 'fusion', 'mhealth', 'swell',
            'usc-had', 'uci-har', 'pamap2', 'opportunity', 'realworld']
palette = 'Paired'


def get_magnitude(x):
    return math.sqrt(math.pow(x[0], 2) + math.pow(x[1], 2) + math.pow(x[2], 2))

data_list = []
for dataset in datasets:
    data = dm.load_dataset(dataset, seq_length=100, gyro=True, preprocess={'type': 'normalize'})
    acc_np = np.concatenate((data.x_acc_train, data.x_acc_test)).flatten()
    gyr_np = np.concatenate((data.x_gyr_train, data.x_gyr_test)).flatten()
    choice = np.random.choice(len(acc_np), min_size)
    acc_df = pd.DataFrame(acc_np[choice], columns=['value'])
    gyr_df = pd.DataFrame(gyr_np[choice], columns=['value'])
    acc_df['sensor'] = 'acc'
    acc_df['dataset'] = dataset
    gyr_df['sensor'] = 'gyr'
    gyr_df['dataset'] = dataset
    data_list.append(acc_df)
    data_list.append(gyr_df)

har_data = pd.concat(data_list)
print(har_data.shape)
print('generating plot...')
sns.violinplot(x='dataset', y='value', hue='sensor', data=har_data, split=True, width=1,
               linewidth=0.5, bw=.0005, palette=palette, scale='width')

sns.despine()
ax.tick_params(labelsize=7)
ax.grid(False)
ax.axes.set(ylim=(-3, 3))
ax.set_frame_on(False)
ax.legend(fontsize='xx-large', shadow=True, bbox_to_anchor=(1.05, 1.05))

plt.show()
