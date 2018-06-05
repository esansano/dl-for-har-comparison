# Plots a heatmap with info from each dataset

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import datamanager as dm
from matplotlib.colors import LinearSegmentedColormap

sns.set_style("whitegrid")
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(111)

left = 0.10      # the left side of the subplots of the figure
right = 0.97     # the right side of the subplots of the figure
bottom = 0.05    # the bottom of the subplots of the figure
top = 0.95       # the top of the subplots of the figure
wspace = 0.05    # the amount of width reserved for blank space between subplots
hspace = 0.05    # the amount of height reserved for white space between subplots

fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

datasets = ['activemiles', 'hhar', 'fusion', 'uci-har',
            'opportunity', 'usc-had', 'realworld', 'mhealth', 'swell', 'pamap2']

summary = dm.get_dataset_summary()
plot_data = np.empty((10, 2), dtype=float)
activities = []
for index, dataset in enumerate(datasets):
    i = summary.index[summary.ix[:, 0] == dataset].tolist()[0]
    acts = [int(x) for x in summary.values[i, 1:] if x > 0]
    activities.append(acts)
    size = sum(acts)
    ratio = size / len(acts)
    plot_data[index, 0] = size / 10
    plot_data[index, 1] = ratio

color = 'Blues'
new_cmap = LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=color, a=0.2, b=0.6),
                                             plt.get_cmap(color)(np.linspace(0.2, 0.6, 100)))

plt.scatter(range(plot_data.shape[0]), plot_data[:, 1], s=plot_data[:, 0], c=new_cmap(0.95), alpha=0.4)
for i in range(plot_data.shape[0]):
    size = sum(activities[i])
    ax.annotate(datasets[i], (i, plot_data[i, 1] + 40 * math.sqrt(size)), size=15, weight='normal', ha='center')

    # piechart
    r = 0
    for activity in activities[i]:
        r = r + activity / size
        x = [0] + np.cos(np.linspace(2 * np.pi * r, 2 * np.pi * r + 0.015, 10)).tolist()
        y = [0] + np.sin(np.linspace(2 * np.pi * r, 2 * np.pi * r + 0.015, 10)).tolist()
        xy1 = np.column_stack([x, y])
        s1 = np.abs(xy1).max()
        ax.scatter(i, plot_data[i, 1], marker=(xy1, 0), s=(s1 ** 2) * plot_data[i, 0],
                   color=new_cmap(0.75), alpha=0.7)

ax.set_ylabel('number of data points per activity', size=14, weight='bold')
ax.set_ylim([-5000, 60000])
ax.set_xlim([-0.6, 10.1])
ax.set_yticks(range(0, 60000, 10000))
ax.set_xticks([])
ax.tick_params(direction='out', length=8, width=3, color='darkgray')
ax.grid(which='both', axis='y', linestyle='')
ax.grid(which='major', axis='x', linestyle='')
ax.set_frame_on(False)

plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()
