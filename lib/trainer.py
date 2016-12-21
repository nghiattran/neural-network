import math


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
            self.learning_rate = setting['rate'] if 'rate' in setting else 0.5
            self.error = setting['error'] if 'error' in setting else 0.05
            self.squash = setting['quash'] if 'quash' in setting else LOGISTIC
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

        init_method = setting['inital'] if setting and 'inital' in setting else None

        self.network.initialize(init_method)

        error = 99
        count = 1
        while error > self.error and count < 60:
            count += 1
            error = 0
            for data in training_set:
                # print()
                value = self.network.activate(data['input'])
                error += sum(value)
                self.network.propagate(data['output'])
            print(error)