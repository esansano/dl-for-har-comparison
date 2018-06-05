import argparse
import json
import os
import torch
import numpy as np
from lab import lab
from utils import datamanager as dm
from utils.exp_log import Logger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset

parser = argparse.ArgumentParser(description='Run experiment.')
parser.add_argument('-e', '--experiment', default='cnn/cnn1_exp', help="experiment definition (json file)")
parser.add_argument('-d', '--dataset', default='hapt', help="from ['activemiles', 'hhar', 'fusion']")
parser.add_argument('-f', '--nfolds', default=5, help="number of folds", type=int)
parser.add_argument('-s', '--save', dest='save', action='store_true')


class Experiment:

    def __init__(self, exp_def_file, dataset, n_folds, save_log):
        self.exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'exp', exp_def_file + '.json')
        self.dataset = dataset
        self.n_folds = n_folds
        self.k_fold = 1
        self.save_log = save_log
        self.logger = None

    @staticmethod
    def __load_data__(dataset, gyro, preprocess):
        return dm.load_dataset(dataset, seq_length=100, gyro=gyro, preprocess=preprocess)

    def update(self, **kwargs):
        return self.logger.update(self.k_fold, **kwargs)

    def run(self):
        with open(self.exp_path, 'r') as exp_file:
            experiment_definition = json.load(exp_file)

        gyro = experiment_definition["gyroscope"]
        arch_type = experiment_definition["type"]
        name = experiment_definition["name"]
        preprocess = experiment_definition["preprocess"]
        log_path = os.path.dirname('{}{}..{}log{}'.format(os.path.dirname(os.path.abspath(__file__)),
                                                          os.sep, os.sep, os.sep))

        self.logger = Logger(exp_name=name, dataset=self.dataset, n_folds=self.n_folds,
                             save_log=self.save_log, log_path=log_path)

        ds = self.__load_data__(self.dataset, gyro=gyro, preprocess=preprocess)

        x, y = np.concatenate((ds.x_acc_train, ds.x_gyr_train), axis=2) if gyro else ds.x_acc_train, ds.y_train
        x_ts, y_ts = np.concatenate((ds.x_acc_test, ds.x_gyr_test), axis=2) if gyro else ds.x_acc_test, ds.y_test

        print("Test: features shape, labels shape, mean, standard deviation")
        print(x_ts.shape, y_ts.shape, np.mean(x_ts), np.std(x_ts))

        if arch_type == 'cnn':
            x_ts = np.reshape(x_ts, newshape=(x_ts.shape[0], 1, x_ts.shape[1], x_ts.shape[2]))

        x_ts = torch.from_numpy(x_ts).type(torch.FloatTensor)
        y_ts = torch.from_numpy(y_ts).type(torch.LongTensor)
        
        n_out = np.unique(y).size
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=0)
        self.k_fold = 1
        for tr_i, va_i in skf.split(X=x, y=y):
            x_tr, x_va = x[tr_i], x[va_i]
            y_tr, y_va = y[tr_i], y[va_i]

            print("Training: features shape, labels shape, mean, standard deviation")
            print(x_tr.shape, y_tr.shape, np.mean(x_tr), np.std(x_tr))
            print("Validation: features shape, labels shape, mean, standard deviation")
            print(x_va.shape, y_va.shape, np.mean(x_va), np.std(x_va))

            if arch_type == 'cnn':
                x_tr = np.reshape(x_tr, newshape=(x_tr.shape[0], 1, x_tr.shape[1], x_tr.shape[2]))
                x_va = np.reshape(x_va, newshape=(x_va.shape[0], 1, x_va.shape[1], x_va.shape[2]))

            x_tr = torch.from_numpy(x_tr).type(torch.FloatTensor)
            y_tr = torch.from_numpy(y_tr).type(torch.LongTensor)
            x_va = torch.from_numpy(x_va).type(torch.FloatTensor)
            y_va = torch.from_numpy(y_va).type(torch.LongTensor)

            print(np.unique(y_tr.numpy(), return_counts=True))
            print(np.unique(y_va.numpy(), return_counts=True))
            print(np.unique(y_ts.numpy(), return_counts=True))

            lab_experiment = lab.build_experiment(self.exp_path, n_out, seed=0)
            print(lab_experiment.model)

            lab_experiment.run(train=TensorDataset(x_tr, y_tr),
                               validation=TensorDataset(x_va, y_va),
                               test=TensorDataset(x_ts, y_ts),
                               update_callback=self.update)
            self.k_fold += 1


if __name__ == "__main__":
    args = parser.parse_args()
    experiment = Experiment(args.experiment, args.dataset, args.nfolds, args.save)
    experiment.run()
