import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()

    def forward(self, x):
        out = x
        for module_item in self.children():
            out = module_item(out)
        return out


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # Reshape x to one dimension
        x = x.view(-1, self.num_flat_features(x))
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def __repr__(self):
        return 'Flatten()'


class Last(nn.Module):
    def __init__(self):
        super(Last, self).__init__()

    def forward(self, x):
        return x[0][:, -1, :]

    def __repr__(self):
        return 'Last()'


class Select(nn.Module):

    def __init__(self, position):
        super(Select, self).__init__()
        self.position = position

    def forward(self, x):
        return x[self.position]

    def __repr__(self):
        return 'Select(position={})'.format(self.position)


class Unsqueeze(nn.Module):

    def __init__(self, axis):
        super(Unsqueeze, self).__init__()
        self.axis = axis

    def forward(self, x):
        return x.unsqueeze(self.axis)

    def __repr__(self):
        return 'Unsqueeze(axis={})'.format(self.axis)



def get_fully_connected_module(module_definition):
    name = module_definition['name']
    input_size = module_definition['input_size']
    output_size = module_definition['output_size']
    fc_layer = nn.Linear(input_size, output_size)
    if 'init' in module_definition:
        if module_definition['init'] == 'xavier':
            torch.nn.init.xavier_uniform(fc_layer.weight)
    return name, fc_layer, output_size


def get_activation_module(module_definition):
    name = module_definition['name']
    return name, __get_activation(module_definition['function']), -1


def get_dropout_module(module_definition):
    name = module_definition['name']
    return name, nn.Dropout(p=module_definition['p']), -1


def get_convolutional_module(module_definition):
    kernel = module_definition['kernel']
    name = module_definition['name']
    if module_definition['type'] == 'conv2d':
        stride = __get_tuple(kernel['stride'])
        kernel_size = __get_tuple(kernel['size'])
        conv_module = nn.Conv2d(kernel['channels'][0],
                                kernel['channels'][1],
                                kernel_size=kernel_size,
                                stride=stride)
        if 'init' in kernel:
            if kernel['init'] == 'xavier':
                torch.nn.init.xavier_uniform(conv_module.weight)
        return name, conv_module, -1


def __get_tuple(value, dim=2, default=1):
    tupled_value = [default] * dim
    if value is not None:
        if type(value) is int:
            tupled_value = [value] * dim
        elif type(value) is list:
            tupled_value = value
    return tuple(tupled_value)


def get_pooling_module(module_definition):
    pool_module = None
    name = module_definition['name']
    if module_definition['type'] == 'max2d':
        stride = __get_tuple(module_definition['stride'])
        kernel_size = __get_tuple(module_definition['size'])
        pool_module = nn.MaxPool2d(kernel_size=kernel_size,
                                   stride=stride)
    elif module_definition['type'] == 'adaptativemax2d':
        pool_module = nn.AdaptiveMaxPool2d(output_size=module_definition['output_size'])

    return name, pool_module, -1


def get_flatten_module(module_definition):
    name = module_definition['name']
    return name, Flatten(), -1


def get_select_module(module_definition):
    name = module_definition['name']
    return name, Select(module_definition['position']), -1


def get_unsqueeze_module(module_definition):
    name = module_definition['name']
    return name, Unsqueeze(module_definition['axis']), -1


def get_lstm_module(module_definition):
    params = __get_rnn_params(module_definition)
    lstm_module = nn.LSTM(**params)
    out_size = params['hidden_size']
    if 'bidirectional' in params and params['bidirectional']:
        out_size = params['hidden_size'] * 2
    return module_definition['name'], lstm_module, out_size


def get_gru_module(module_definition):
    params = __get_rnn_params(module_definition)
    gru_module = nn.GRU(**params)
    return module_definition['name'], gru_module, params['hidden_size']


def __get_rnn_params(module_definition):
    dropout = 0
    bias = True
    batch_first = False
    bidirectional = False
    input_size = module_definition['input_size']
    hidden_size = module_definition['hidden_size']
    num_layers = module_definition['num_layers']
    if 'bias' in module_definition:
        bias = module_definition['bias']
    if 'batch_first' in module_definition:
        batch_first = module_definition['batch_first']
    if 'dropout' in module_definition:
        dropout = module_definition['dropout']
    if 'bidirectional' in module_definition:
        bidirectional = module_definition['bidirectional']

    return {'input_size': input_size, 'hidden_size': hidden_size,
            'num_layers': num_layers, 'bias': bias,
            'batch_first': batch_first,
            'dropout': dropout, 'bidirectional': bidirectional}


def get_last_value_module(module_definition):
    name = module_definition['name']
    return name, Last(), -1


def get_criterion(criterion_definition):
    crit_type = criterion_definition['type'].lower()
    params = criterion_definition['parameters']
    criterion = nn.CrossEntropyLoss()
    if crit_type == 'mse':
        criterion = nn.MSELoss(**params)
    elif crit_type == 'ce':
        if 'weight' in params:
            if torch.cuda.is_available():
                params['weight'] = torch.from_numpy(np.array(params['weight'])).float().cuda()
            else:
                params['weight'] = torch.from_numpy(np.array(params['weight'])).float()
        criterion = nn.CrossEntropyLoss(**params)
    elif crit_type == 'nll':
        criterion = nn.NLLLoss(**params)
    elif crit_type == 'mml':
        criterion = nn.MultiMarginLoss(**params)

    return criterion


def get_optimizer(optimizer_definition, model, lr):
    opt_type = optimizer_definition['type'].lower()
    params = optimizer_definition['parameters']
    if opt_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, **params)
    elif opt_type == 'sgdm':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, **params)
    elif opt_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, **params)
    elif opt_type == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, **params)
    elif opt_type == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr, **params)
    elif opt_type == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr=lr, **params)
    elif opt_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, **params)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr, **params)
    return optimizer


def __get_activation(ac_function):
    if ac_function == 'sigmoid':
        return nn.Sigmoid()
    elif ac_function == 'tanh':
        return nn.Tanh()
    elif ac_function == 'relu':
        return nn.ReLU()
    elif ac_function == 'selu':
        return nn.SELU()
    elif ac_function == 'relu6':
        return nn.ReLU6()
    elif ac_function == 'elu':
        return nn.ELU()
    elif ac_function == 'leaky_relu':
        return nn.LeakyReLU()
    return nn.ReLU()
