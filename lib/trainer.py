import math
import random


def LOGISTIC(x, derivative = False):
    if derivative == True:
        fx = LOGISTIC(x)
        return fx * (1 - fx)
    return 1 / (1 + pow(math.e, -1 * x))

class Trainer(object):
    def __init__(self, network = None, setting = None):
        if network is not None:
            self.set_network(network)
        else:
            self.network = None

        self.set({})

    def set(self, setting):
        if type(setting) is dict:
            self.learning_rate = setting['rate'] if 'rate' in setting else 0.1
            self.error = setting['error'] if 'error' in setting else 0.005
            self.squash = setting['quash'] if 'quash' in setting else LOGISTIC
            self.momentum = setting['momentum'] if 'momentum' in setting else 0.95
            if not (0 <= self.momentum and self.momentum < 1):
                raise ValueError('Momemtum value has to be: 0 <= momemtum < 1')
        else:
            raise ValueError('The second argument must be a dictionary')

    def set_network(self, network):
        self.network = network
        self.network.set_trainer(self)

    def train(self, training_set, setting = None):
        if self.network is None:
            raise ValueError('Network has to be set before trainning.')

        if setting is not None:
            self.set(setting)

        epoch_limit = setting['epoch'] if setting and 'epoch' in setting else 5000
        log = setting['log'] if setting and 'log' in setting else 0
        update = setting['update'] if setting and 'update' in setting else None
        shuffle = setting['shuffle'] if setting and 'shuffle' in setting else False
        error = 1
        epoch = 0

        while error > self.error and epoch < epoch_limit:
            epoch += 1
            previous_error = error
            error = 0
            for data in training_set:
                self.network.activate(data['input'])
                errors = self.network.propagate(data['output'])
                error += pow(sum(errors), 2)

            if log != 0 and epoch % log == 0:
                print(epoch, error, self.learning_rate)

            if update is not None:
                self.learning_rate = update(previous_error, error, self.learning_rate)

            if shuffle:
                random.shuffle(training_set)
        return error, epoch

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