import os
import numpy as np
import pandas as pd
from utils import dataimport as di


build_dict = {'activemiles': di.build_activemiles, 'hhar': di.build_hhar, 'swell': di.build_swell,
              'fusion': di.build_fusion, 'usc-had': di.build_usc_had, 'mhealth': di.build_mhealth,
              'uci-har': di.build_uci_har, 'pamap2': di.build_pamap2, 'opportunity': di.build_opportunity,
              'realworld': di.build_realworld}


def load_dataset(dataset, seq_length, gyro, preprocess):
    ds = di.load_saved_data(dataset_name=dataset, seq_length=seq_length, gyro=gyro, preprocess=preprocess)
    if ds is None:
        build_dict[dataset](seq_length=seq_length)
        ds = di.load_saved_data(dataset_name=dataset, seq_length=seq_length, gyro=gyro, preprocess=preprocess)
    return ds


def get_dataset_summary():
    summary_file_name = 'dataset_summary.txt'
    if os.path.isfile(summary_file_name):
        summary = pd.read_csv(summary_file_name)
    else:
        summary_np = np.zeros((10, 12))
        index = []
        columns = ['dp_activity_{}'.format(i + 1) for i in range(12)]
        for i, dataset in enumerate(build_dict):
            data = load_dataset(dataset, seq_length=100, gyro=True, preprocess={'type': 'standardize'})
            acts = np.unique(np.concatenate((data.y_train, data.y_test)), return_counts=True)[1].tolist()
            acts.extend([0] * (12 - len(acts)))
            index.append(dataset)
            summary_np[i, :] = acts
        summary = pd.DataFrame(summary_np, columns=columns, index=index)
        summary.to_csv(summary_file_name)
    return summary
