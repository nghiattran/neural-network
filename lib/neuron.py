from lib.connection import Connection
import math


class Neuron(object):
    __id_count__ = 0

    @staticmethod
    def generate_id():
        Neuron.__id_count__ += 1
        return Neuron.__id_count__

    def __init__(self, layer, setting = None):
        self.previous = []
        self.next = []
        self.layer = layer
        self.trainer = None
        self.error_gradient = 0

        if setting is None:
            self.id = Neuron.generate_id()
            self.activation = 0
            self.threshold = 0
            self.state = 0
            self.old = 0
            self.squash = LOGISTIC
        else:
            try:
                self.id = setting['id']
                self.activation = setting['activation']
                self.threshold = setting['threshold']
                self.state = setting['state']
                self.old = setting['old']
                self.squash = {
                    'LOGISTIC': LOGISTIC,
                    'TANH': TANH,
                    'LINEAR': LINEAR,
                    'RELU': RELU
                }[setting['squash']]
            except:
                raise ValueError('Input file is corrupted.')

    def initialize(self, cb = None):
        self.threshold = cb()
        return self

    def to_json(self):
        return {
            'id': self.id,
            'activation': self.activation,
            'state': self.state,
            'old': self.old,
            'threshold': self.threshold,
            'squash': self.squash.__name__
        }

    @staticmethod
    def from_json(layer, setting):
        return Neuron(layer, setting)

    def activate(self, input = None):
        if input is not None:
            self.activation = input
            return self.activation

        self.old = self.state
        self.state = 0
        for connection in self.previous:
            self.state += connection.get_state()

        # eq 6.2
        self.activation = self.squash(self.state - self.threshold)

        return self.activation

    def propagate(self, learning_rate, output = None, momentum = 0):
        # output is None means this is a neuron in hidden layer
        if output is None:
            # eq 6.15
            error = 0
            for connection in self.next:
                error += connection.get_error()
        else:
            # eq 6.4
            error = output - self.activation

        # eq 6.13
        self.error_gradient = self.squash(self.state - self.threshold, True) * error

        # Update all connections
        for connection in self.previous:
            connection.calculate_weight_correction(learning_rate, momentum)

        for connection in self.next:
            connection.update()

        # Update threshold
        self.threshold += (-1) * learning_rate * self.error_gradient

        return error

    def connect(self, next_neuron, weight = 0):
        Connection(self, next_neuron, weight)
        return self


def LOGISTIC(x, derivative = False):
    if derivative:
        fx = LOGISTIC(x)
        return fx * (1 - fx)

    return 1 / (1 + pow(math.e, -1 * x))


def TANH(x, derivative = False):
    if derivative:
        fx = TANH(x)
        return 1 - math.pow(fx, 2)

    p = math.exp(x)
    n = 1 / p
    return (p - n) / (p + n)


def RELU(x, derivative = False):
    if derivative:
        return 1 if x > 0 else 0

    return x if x > 0 else 0

def LINEAR(x, derivative = False):
    if derivative:
        return 1
    return x

