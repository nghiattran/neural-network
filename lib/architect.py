# from lib.layer import Layer as Layer
from .layer import Layer
from .trainer import Trainer

class Perceptron(object):
    def __init__(self, setting):
        if 'input' not in setting or 'output' not in setting:
            raise Exception('Input layer or output layer for both are missing')

        if type(setting['input']) is not Layer or type(setting['output']) is not Layer:
            raise ValueError('Input layer, or output layer, or both are not Layer instances')

        self.input = setting['input']
        self.output = setting['output']

        if 'hidden' in setting:
            self.hidden = setting['hidden']
        else:
            self.hidden = []

    def set_trainer(self, trainer):
        if type(trainer) is not Trainer:
            raise ValueError('trainer must be a Trainer instance')
        self.input.set_trainer(trainer)
        for i in range(len(self.hidden)):
            self.hidden[i].set_trainer(trainer)
        self.output.set_trainer(trainer)

    def initialize(self):
        self.input.initialize()
        for i in range(len(self.hidden)):
            self.hidden[i].initialize()

    def get_layers(self):
        layers =  [self.input, self.output]
        layers[1:1] = self.hidden
        return layers

    def activate(self, input):
        self.input.activate(input)
        for i in range(len(self.hidden)):
            self.hidden[i].activate()
        self.output.activate()

        return self.get_outputs()

    def get_outputs(self):
        return self.output.get_activations()

    def train_weight(self):
        self.output.train_weight()
        for i in range(len(self.hidden) - 1, -1, -1):
            self.hidden[i].train_weight()
        self.input.train_weight()

    def get_connections(self):
        connections = self.input.get_connections()
        for i in range(len(self.hidden)):
            connections += self.hidden[i].get_connections()
        return  connections

