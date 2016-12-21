from lib.connection import Connection
import random


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
        else:
            try:
                self.id = setting['id']
                self.activation = setting['activation']
                self.threshold = setting['threshold']
                self.state = setting['state']
                self.old = setting['old']
            except:
                raise ValueError('Input file is corrupted.')

    def initialize(self, cb = None):
        if cb is None:
            self.threshold = random.uniform(-0.5, 0.5)
            return self

        self.threshold = cb()
        return self

    def to_json(self):
        return {
            'id': self.id,
            'activation': self.activation,
            'state': self.state,
            'old': self.old,
            'threshold': self.threshold
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
        self.activation = self.trainer.squash(self.state - self.threshold)

        return self.activation

    def propagate(self, output = None):
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
        self.error_gradient = self.trainer.squash(self.state - self.threshold, True) * error

        for connection in self.previous:
            connection.calculate_weight_correction(0.1)

        for connection in self.next:
            connection.update()

    def set_trainer(self, trainer):
        self.trainer = trainer
        return self

    def connect(self, next_neuron, weight = 0):
        Connection(self, next_neuron, weight)
        return self