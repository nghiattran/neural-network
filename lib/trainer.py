import math
import random
import types


class Trainer(object):
    def __init__(self, network = None):
        if network is not None:
            self.set_network(network)
        else:
            self.network = None

    def set_network(self, network):
        self.network = network

    def train(self, training_set, setting = None):
        # Validate all parameters
        if self.network is None:
            raise ValueError('Network has to be set before trainning.')

        if setting is None:
            setting = {}
        elif type(setting) is not dict:
            raise ValueError('The second argument must be a dictionary')

        # Configure settings
        epoch_limit = setting['epoch'] if 'epoch' in setting else 5000
        log = setting['log'] if 'log' in setting else 0
        shuffle = setting['shuffle'] if 'shuffle' in setting else False
        rate = setting['rate'] if 'rate' in setting else 0.1
        error = setting['error'] if 'error' in setting else 0.005
        momentum = setting['momentum'] if 'momentum' in setting else 0.95
        if not (0 <= momentum and momentum < 1):
            raise ValueError('Momemtum value has to be: 0 <= momemtum < 1')

        if type(rate) is float:
            learning_rate = rate
            rate = None
        elif isinstance(rate, types.FunctionType):
            learning_rate = rate()
        else:
            raise ValueError('Rate value must be a float or a function')

        # Train
        sum_error = 1
        epoch = 0
        while sum_error > error and epoch < epoch_limit:
            epoch += 1
            previous_error = sum_error
            sum_error = 0
            for data in training_set:
                self.network.activate(data['input'])
                errors = self.network.propagate(
                    learning_rate=learning_rate,
                    momentum=momentum,
                    outputs=data['output'])
                sum_error += pow(sum(errors), 2)

            if log != 0 and epoch % log == 0:
                print(epoch, sum_error, learning_rate)

            if rate is not None:
                learning_rate = rate(previous_error, sum_error, learning_rate)

            if shuffle:
                random.shuffle(training_set)
        return sum_error, epoch

    def XOR(self, settings = None):
        if settings is None:
            settings = {
                'shuffle': True,
                'momentum': 0.99,
            }

        return self.train([{
            'input': [0, 0],
            'output': [0]
        }, {
            'input': [0, 1],
            'output': [1]
        }, {
            'input': [1, 0],
            'output': [1]
        },{
            'input': [1, 1],
            'output': [0]
        }], settings)

    def AND(self, settings = None):
        if settings is None:
            settings = {
                'shuffle': True,
                'momentum': 0.99,
            }

        return self.train([{
            'input': [0, 0],
            'output': [0]
        }, {
            'input': [0, 1],
            'output': [0]
        }, {
            'input': [1, 0],
            'output': [0]
        }, {
            'input': [1, 1],
            'output': [1]
        }], settings)