import json
import torch
from lab import models as m, experiments as e

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


def build_experiment(experimentfile, out_size=None, seed=None):
    experiment_definition = __load_json(experimentfile)
    architecture_definition = __load_json(experiment_definition['architecture_file'])
    model = __build_model(architecture_definition, out_size, seed)
    experiment = e.Experiment(model, experiment_definition)
    return experiment


def __build_model(architecture_definition, out_size, seed=None):
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
