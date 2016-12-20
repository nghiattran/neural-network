# from lib.layer import Layer as Layer
from .layer import Layer
from .trainer import Trainer

class Perceptron(object):
    def __init__(self, setting = None):
        if setting is None:
            return

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

        self.input.set_layer('input')
        self.output.set_layer('output')
        for i in range(len(self.hidden)):
            self.hidden[i].set_layer(i)

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

    def propagate(self):
        self.output.propagate()
        for i in range(len(self.hidden) - 1, -1, -1):
            self.hidden[i].propagate()
        self.input.propagate()

        self.output.update()
        for i in range(len(self.hidden) - 1, -1, -1):
            self.hidden[i].update()
        self.input.update()

    def get_connections(self):
        connections = [connection for connection in self.input.get_connections()]
        for i in range(len(self.hidden)):
            connections += self.hidden[i].get_connections()
        return  connections

    def get_neurons(self):
        neurons = [neuron for neuron in self.input.get_neurons()]
        for i in range(len(self.hidden)):
            neurons += self.hidden[i].get_neurons()
        neurons += self.output.get_neurons()
        return neurons

    def to_json(self):
        return {
            'input': self.input.to_json(),
            'hidden': [hidden.to_json() for hidden in self.hidden],
            'output': self.output.to_json()
        }

    @staticmethod
    def from_json(json):
        return Perceptron().init(json)

    def init(self, json):
        self.input = Layer()
        self.input.init(json['input'])

        if 'hidden' in json:
            self.hidden = []
            for hidden in json['hidden']:
                layer = Layer()
                layer.init(hidden)
                self.hidden.append(layer)

        self.output = Layer()
        self.output.init(json['output'])
        return self
