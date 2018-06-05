import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorchlab import models as m
from torch.autograd import Variable


class Experiment:

    def __init__(self, model, experiment_definition):
        self.model = model
        self.settings = experiment_definition['settings']
        if torch.cuda.is_available():
            self.model.cuda()
        self.criterion = m.get_criterion(experiment_definition['criterion'])
        self.learning_rate = experiment_definition['learning_rate']
        self.optimizer = m.get_optimizer(experiment_definition['optimizer'],
                                         self.model,
                                         self.learning_rate)
        self.batch_size = experiment_definition['batch_size']

    def run(self, train, validation=None, test=None, update_callback=None):
        max_epochs = self.settings['max_epochs']
        loss_validation = 0
        predicted = None
        truth = None
        should_stop = False
        run_test = False

        train = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if validation is not None:
            validation = DataLoader(dataset=validation, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if test is not None:
            test = DataLoader(dataset=test, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if update_callback is not None:
            update_callback(begin=True, max_epochs=max_epochs)

        for epoch in range(max_epochs):
            loss_train = self.train(train)
            if validation is not None:
                loss_validation, predicted, truth = self.evaluate(validation)
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

            if run_test and test is not None:
                kwargs = {'test': True}
                _, predicted, truth = self.evaluate(test)
                if 'prediction' in self.settings['update']:
                    kwargs['prediction'] = predicted
                if 'truth' in self.settings['update']:
                    kwargs['truth'] = truth
                _, _ = update_callback(**kwargs)

    def train(self, dataset):
        total_loss = 0
        n = 0
        self.model.train()

        # Train the Model
        for i, (observations, labels) in enumerate(dataset):
            n += labels.shape[0]
            if torch.cuda.is_available():
                observations = Variable(observations.cuda())
                labels = Variable(labels.cuda())
            else:
                observations = Variable(observations)
                labels = Variable(labels)

            # Forward + Backward + Optimize
            self.optimizer.zero_grad()
            outputs = self.model(observations)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.data[0]

        return total_loss / n

    def evaluate(self, dataset):
        total_loss = 0
        n = 0
        prediction = []
        truth = []
        self.model.eval()

        for i, (observations, labels) in enumerate(dataset):
            n += labels.shape[0]
            if torch.cuda.is_available():
                observations = Variable(observations.cuda())
                labels = Variable(labels.cuda())
            else:
                observations = Variable(observations)
                labels = Variable(labels)
            outputs = self.model(observations)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            if torch.cuda.is_available():
                prediction.extend(predicted.cpu().numpy().tolist())
                truth.extend(labels.cpu().data.numpy().tolist())
            else:
                prediction.extend(predicted.numpy().tolist())
                truth.extend(labels.data.numpy().tolist())

            total_loss += loss.data[0]

        return total_loss / n, np.array(prediction), truth

    def __str__(self):
        return 'Model: \n%s\n Optimizer: %s\n Criterion: %s\n Batch size: %d\n Learning rate: %f' % \
               (self.model, self.optimizer.__class__.__name__, self.criterion.__class__.__name__,
                self.batch_size, self.learning_rate)
