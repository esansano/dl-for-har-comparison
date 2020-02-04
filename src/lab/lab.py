import json
import os
import sys
import time
import math
import copy
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dbn.tf_models import SupervisedDBNClassification
from lab import models as m
from utils.exp_log import Color
from abc import abstractmethod

module_dict = {'activation': m.get_activation_module,
               'dropout': m.get_dropout_module,
               'fully_connected': m.get_fully_connected_module,
               'conv2d': m.get_convolutional_module,
               'max2d': m.get_pooling_module,
               'adaptativemax2d': m.get_pooling_module,
               'flatten': m.get_flatten_module,
               'lstm': m.get_lstm_module,
               'gru': m.get_gru_module,
               'last_value': m.get_last_value_module,
               'select': m.get_select_module,
               'unsqueeze': m.get_unsqueeze_module}


class Trainer:

    def __init__(self, model, experiment_definition):
        self.model = model
        self.settings = experiment_definition['settings']
        if torch.cuda.is_available():
            self.model.cuda()
        self.criterion = m.get_criterion(experiment_definition['criterion'])
        self.learning_rate = experiment_definition['learning_rate']
        self.batch_size = experiment_definition['batch_size']
        self.optimizer = None

    @abstractmethod
    def train(self, train_data, validation_data=None, test_data=None, update_callback=None):
        raise NotImplementedError


class TrainerANN(Trainer):

    def __init__(self, model, experiment_definition):
        super().__init__(model, experiment_definition)
        self.optimizer = m.get_optimizer(experiment_definition['optimizer'],
                                         self.model,
                                         self.learning_rate)

    def train(self, train_data, validation_data=None, test_data=None, update_callback=None):
        max_epochs = self.settings['max_epochs']
        loss_validation = 0
        predicted = None
        truth = None
        should_stop = False
        run_test = False

        dl_trn = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if validation_data is not None:
            dl_val = DataLoader(dataset=validation_data, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if test_data is not None:
            dl_tst = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if update_callback is not None:
            update_callback(begin=True, max_epochs=max_epochs)

        for epoch in range(max_epochs):
            loss_train = self.__train_epoch(dl_trn)
            if validation_data is not None:
                loss_validation, predicted, truth = self.evaluate(dl_val)
            if update_callback is not None:
                kwargs = {'validation': True}
                if 'epoch' in self.settings['update']:
                    kwargs['epoch'] = epoch + 1
                if 'early_stop' in self.settings:
                    kwargs['early_stop'] = self.settings['early_stop']
                if 'loss_train' in self.settings['update']:
                    kwargs['loss_train'] = loss_train
                if 'loss_validation' in self.settings['update']:
                    kwargs['loss_validation'] = loss_validation
                if 'prediction' in self.settings['update']:
                    kwargs['prediction'] = predicted
                if 'truth' in self.settings['update']:
                    kwargs['truth'] = truth
                should_stop, run_test = update_callback(**kwargs)
            if should_stop:
                break

            if run_test and dl_tst is not None:
                kwargs = {'test': True}
                _, predicted, truth = self.evaluate(dl_tst)
                if 'prediction' in self.settings['update']:
                    kwargs['prediction'] = predicted
                if 'truth' in self.settings['update']:
                    kwargs['truth'] = truth
                _, _ = update_callback(**kwargs)

    def __train_epoch(self, dataset):
        total_loss = 0
        n = 0
        self.model.train()

        # Train the Model
        for i, (observations, labels) in enumerate(dataset):
            n += labels.shape[0]

            # Forward + Backward + Optimize
            self.optimizer.zero_grad()
            outputs = self.model(observations)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / n

    def evaluate(self, dataset):
        total_loss = 0
        n = 0
        prediction = []
        truth = []
        self.model.eval()

        for i, (observations, labels) in enumerate(dataset):
            n += labels.shape[0]
            outputs = self.model(observations)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            prediction.extend(predicted.cpu().numpy().tolist())
            truth.extend(labels.cpu().data.numpy().tolist())

            total_loss += loss.item()

        return total_loss / n, np.array(prediction), truth

    def __str__(self):
        return 'Model: \n%s\n Optimizer: %s\n Criterion: %s\n Batch size: %d\n Learning rate: %f' % \
               (self.model, self.optimizer.__class__.__name__, self.criterion.__class__.__name__,
                self.batch_size, self.learning_rate)


class TrainerRBM:

    def __init__(self, rbm, data_loader, device=torch.device('cpu')):
        self.loader = data_loader
        self.rbm = rbm
        self.device = device

    def train(self, max_epochs, early_stop=0, learning_rate=1e-2, k=1, increase_to_cd_k=False,
              lr_decay=True, callback=None, save_log=False, save_model=False, verbose=True):

        batch_size = self.loader.batch_size
        best_error = float('inf')
        best_epoch = 1

        for epoch in range(1, max_epochs + 1):
            start_epoch = time.time()
            epoch_error = 0

            if increase_to_cd_k:
                sampling_steps = int(math.ceil((epoch / max_epochs) * k))
            else:
                sampling_steps = k

            if lr_decay:
                lr = learning_rate / epoch
            else:
                lr = learning_rate

            for x, _ in tqdm(self.loader, leave=False, file=sys.stdout, ncols=70):
                # positive phase
                pos_hidden_probs, pos_hidden_act = self.rbm.to_hidden(x)
                pos_associations = torch.matmul(x.t(), pos_hidden_act)

                # negative phase
                hidden_activations = pos_hidden_act
                visible_probs, hidden_probs = None, None
                for i in range(sampling_steps):
                    visible_probs, _ = self.rbm.to_visible(hidden_activations)
                    hidden_probs, hidden_activations = self.rbm.to_hidden(visible_probs)
                neg_visible_probs = visible_probs
                neg_hidden_probs = hidden_probs
                neg_associations = torch.matmul(neg_visible_probs.t(), neg_hidden_probs)

                g = pos_associations - neg_associations
                grad_update = g / batch_size
                v_bias_update = torch.sum(x - neg_visible_probs, dim=0) / batch_size
                h_bias_update = torch.sum(pos_hidden_probs - neg_hidden_probs, dim=0) / batch_size

                self.rbm.weights.data += lr * grad_update.t()
                self.rbm.v_bias.data += lr * v_bias_update
                self.rbm.h_bias.data += lr * h_bias_update

                # Compute reconstruction error
                epoch_error += torch.sum(torch.sum((x - neg_visible_probs) ** 2, dim=0)).item()

            epoch_error /= len(self.loader.dataset)
            col = Color.default
            if epoch_error < best_error:
                best_error = epoch_error
                best_epoch = epoch
                col = Color.blue

            time_epoch = time.time() - start_epoch

            row = f' {epoch:04d}{col}{epoch_error:8.4f}{Color.default}'
            row += f'   [{best_error:8.4f} ({best_epoch:03d})]'
            row += f'{time_epoch:8.3f}s'

            print(row)

            if 0 < early_stop <= epoch - best_epoch or epoch == max_epochs:
                if verbose:
                    print(f'Best error: {best_error:5.2f} at epoch: {best_epoch}')
                break


class TrainerDBNFT:

    def __init__(self, dbn_ft, train_dataset, validation_dataset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.dbn_ft = dbn_ft
        self.device = device

    def train(self, optimizer, max_epochs=1, early_stop=0, batch_size=64,
              callback=None, save_log=False, save_model=False, verbose=True):

        tr_dldr = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        vl_dldr = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        criterion = m.get_criterion({"type": "ce", "parameters": {}}).to(self.device)
        log_header = {
            'data_params': {'name': 'name'},
            'model_params': self.dbn_ft.layer_sizes,
            'train_params': {'max_epochs': max_epochs, 'early_stop': early_stop, 'batch_size': batch_size}
        }
        logger = Logger(log_header, path='../logs')
        model_name = f'../models/{type(self.dbn_ft).__name__}_{logger.timestamp}.pt'
        best_epoch_tr = 1
        best_epoch_vl = 1
        best_loss_tr = float('inf')
        best_loss_vl = float('inf')
        best_model = copy.deepcopy(self.dbn_ft)

        for epoch in range(1, max_epochs + 1):
            start_epoch = time.time()
            log_epoch = {'epoch': epoch}
            self.dbn_ft.train()
            epoch_loss_tr = 0
            epoch_loss_vl = 0

            for x, y in tqdm(tr_dldr, leave=False, file=sys.stdout, ncols=70):
                optimizer.zero_grad()
                y_hat = self.dbn_ft(x)
                # loss_tr = binary_cross_entropy(y_hat, y, reduction='sum')
                # print(y_hat.size(), y.size())
                loss_tr = criterion(y_hat, y)
                epoch_loss_tr += loss_tr.item()
                loss_tr.backward()
                optimizer.step()

            with torch.no_grad():
                hits = 0
                self.dbn_ft.eval()
                total_loss = 0
                n = 0
                prediction = []
                truth = []
                for x, y in tqdm(vl_dldr, leave=False, file=sys.stdout, ncols=70):
                    y_hat = self.dbn_ft.forward(x)
                    y_np = y_hat.cpu().numpy()
                    loss_vl = criterion(y_hat, y)
                    epoch_loss_vl += loss_vl.item()

                    _, y_hat = torch.max(y_hat, 1)

                    prediction.extend(y_hat.cpu().numpy().tolist())
                    truth.extend(y.cpu().data.numpy().tolist())

            acc = accuracy_score(prediction, truth)

            epoch_loss_tr /= len(tr_dldr.dataset)
            epoch_loss_vl /= len(vl_dldr.dataset)

            end_epoch = time.time()
            time_epoch = end_epoch - start_epoch

            log = f'{epoch:4d}/{max_epochs:3d} '
            log += f'{time_epoch:6.3f}s'
            coltr = Color.default

            if epoch_loss_tr < best_loss_tr:
                best_loss_tr = epoch_loss_tr
                coltr = Color.magenta
                best_epoch_tr = epoch

            log += f'| {coltr}{epoch_loss_tr:10.6f}{Color.default} '

            log_epoch['epoch_loss_tr'] = epoch_loss_tr

            colvl = Color.default
            if epoch_loss_vl < best_loss_vl:
                best_loss_vl = epoch_loss_vl
                colvl = Color.green
                best_epoch_vl = epoch
                best_model = copy.deepcopy(self.dbn_ft)

            log += f'| {colvl}{epoch_loss_vl:10.6f}{Color.default} | {acc:6.4f} | '

            log_epoch['epoch_loss_vl'] = epoch_loss_vl

            logger.item(log_epoch)

            if save_log:
                logger.save()

            if verbose:
                print(log)

            if save_model and best_epoch_vl == epoch:
                torch.save(best_model.state_dict(), model_name)

            if callback is not None and best_epoch_vl == epoch:
                callback(epoch, self.dbn_ft, best_model)

            if 0 < early_stop <= epoch - best_epoch_vl or epoch == max_epochs:
                if verbose:
                    print(f'Best loss on training data: {best_loss_tr:9.7f} at epoch: {best_epoch_tr}')
                    print(f'Best loss on validation data: {best_loss_vl:9.7f} at epoch: {best_epoch_vl}')
                break

        return best_model


class Logger:

    def __init__(self, header, path, name='log', timestamp=True):
        self.log_data = {
            'header': header,
            'log': []
        }
        if timestamp:
            self.timestamp = int(time.time())
            self.name = f'{path}/{name}_{self.timestamp}.json'
        else:
            self.name = f'{path}/{name}.json'

    def item(self, log_item):
        self.log_data['log'].append(log_item)

    def save(self):
        with open(self.name, 'w') as log_file:
            json.dump(self.log_data, log_file)


class RBMTransform:

    def __init__(self, trained_rbms):
        self.trained_rbms = trained_rbms

    def __call__(self, sample):
        x = sample
        for rbm in self.trained_rbms:
            _, x = rbm.to_hidden(x)
        return x


class TrainerDBN(Trainer):

    def __init__(self, model, experiment_definition, device):
        super().__init__(model, experiment_definition)
        self.model = model
        self.experiment_definition = experiment_definition
        self.device = device

    def train(self, train_data, validation_data=None, test_data=None, update_callback=None):
        max_epochs = self.experiment_definition['settings']['max_epochs']
        early_stop = self.experiment_definition['settings']['early_stop']
        lr_decay = self.experiment_definition['settings']['lr_decay']
        k = self.experiment_definition['settings']['k']
        increase_to_cd_k = self.experiment_definition['settings']['increase_to_cd_k']
        batch_size = self.experiment_definition['batch_size']
        learning_rate = self.experiment_definition['learning_rate']
        loss_validation = 0
        predicted = None
        truth = None
        should_stop = False
        run_test = False

        # dl_trn = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=False)
        #
        # if validation_data is not None:
        #     dl_val = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=True, drop_last=False)
        #
        # if test_data is not None:
        #     dl_tst = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=False)

        if update_callback is not None:
            update_callback(begin=True, max_epochs=max_epochs)

        trained_rbms = []
        train_data_ds = m.DBNDataset(train_data)
        for i, rbm in enumerate(self.model.rbms):
            rbm.to(self.device)
            train_data_ds.transform = RBMTransform(trained_rbms)
            loader = DataLoader(train_data_ds, batch_size=batch_size, shuffle=True, drop_last=False)
            rbm_trainer = TrainerRBM(rbm, loader)
            print(f'Training RBM {i + 1} [visible units: {rbm.visible_units}, hidden units: {rbm.hidden_units}]')
            rbm_trainer.train(max_epochs=max_epochs, early_stop=early_stop, learning_rate=learning_rate,
                              k=k, increase_to_cd_k=increase_to_cd_k, lr_decay=lr_decay)
            trained_rbms.append(rbm)

        dbn_ft = m.DBNFT(trained_dbn=self.model, output_size=self.model.out_size).to(self.device)
        train_dataset = m.DBNDataset(train_data)
        validation_dataset = m.DBNDataset(validation_data)
        trainer = TrainerDBNFT(dbn_ft, train_dataset, validation_dataset)

        learning_rate = 1e-3
        weight_decay = 1e-6
        optimizer = torch.optim.Adam(dbn_ft.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(dbn_ft.parameters(), lr=learning_rate, weight_decay=weight_decay)

        print('Fine-tuning DBN')
        print(dbn_ft)
        trained_model = trainer.train(optimizer=optimizer, max_epochs=100, early_stop=10, batch_size=batch_size,
                                      callback=None, save_log=False, save_model=False, verbose=True)


def build_experiment(experimentfile, out_size=None, seed=None, device=torch.device("cpu")):
    experiment_def = __load_json(experimentfile)
    arq_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', experiment_def['architecture_file'])
    architecture_def = __load_json(arq_file)
    if architecture_def['type'] == 'dbn':
        model = __build_model_dbn(architecture_def, experiment_def)
        return model
    else:
        model = __build_model(architecture_def, out_size, seed, device)
        trainer = TrainerANN(model, experiment_def)
    return trainer


def __build_model_dbn(architecture_definition, experiment_definition):

    rbms = []
    act = architecture_definition['modules'][0]['act']

    for module in architecture_definition['modules']:
        rbms.append(module['size'])
    learning_rate_rbm = experiment_definition["learning_rate_rbm"]
    learning_rate = experiment_definition["learning_rate"]
    batch_size = experiment_definition["batch_size"]
    patience_rbm = experiment_definition["patience_rbm"]
    patience_ft = experiment_definition["patience_ft"]
    max_epochs_rbm = experiment_definition["max_epochs_rbm"]
    max_epochs_ft = experiment_definition["max_epochs_ft"]

    model = SupervisedDBNClassification(hidden_layers_structure=rbms,
                                        learning_rate_rbm=learning_rate_rbm,
                                        learning_rate=learning_rate,
                                        max_epochs_rbm=max_epochs_rbm,
                                        patience_rbm=patience_rbm,
                                        max_epochs_ft=max_epochs_ft,
                                        patience_ft=patience_ft,
                                        batch_size=batch_size,
                                        activation_function=act,
                                        dropout_p=0.0)

    return model


def __build_model(architecture_definition, out_size, seed=None, device=torch.device("cpu")):
    if seed is not None:
        torch.manual_seed(seed)
    last_out_size = 0
    model = m.NNModel()
    for module_def in architecture_definition['modules']:
        name, module, layer_out_size = module_dict[module_def['type']](module_def)
        model.add_module(name, module)
        if layer_out_size > 0:
            last_out_size = layer_out_size

    if out_size is not None:
        module = torch.nn.Linear(last_out_size, out_size)
        model.add_module('output', module)
    model.to(device)

    return model


def __load_json(filepath):
    if filepath[-5:] != '.json':
        filepath = filepath + '.json'
    with open(filepath, 'r') as exp_file:
        return json.load(exp_file)


def save_model(model, name, path='', log=False):
    with open(path + name + '.ptm', 'wb') as f:
        torch.save(model, f)
    if log:
        print('Model saved to %s%s' % (path, name))


def load_model(name, path='', log=False):
    model = torch.load(path + name, map_location=lambda storage, loc: storage)
    if log:
        print('Model loaded from %s%s' % (path, name))
    if torch.cuda.is_available():
        model.cuda()
    return model
