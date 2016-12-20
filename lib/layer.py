from lib.neuron import Neuron
import zipfile

class Layer(object):
    def __init__(self, setting = None):
        self.neurons = []
        if setting is None:
            return

        if type(setting) is int:
            for i in range(setting):
                self.neurons.append(Neuron(self))
        else:
            raise ValueError('Layer constructor only takes integer argument')

    def get_neurons(self):
        return self.neurons

    def project(self, layer):
        if type(layer) is not Layer:
            raise ValueError('Projected object is not a Layer instance')

        neurons = layer.get_neurons()
        for neuron in self.neurons:
            for next_neuron in neurons:
                neuron.connect(next_neuron)
        return self

    def propagate(self):
        for i in range(len(self.neurons) - 1, -1, -1):
            self.neurons[i].propagate()

    def activate(self, input = None):
        activation = []

        if input is not None:
            print(self.neurons)
            if len(input) != len(self.neurons):
                raise ValueError('Input size does not match number of neurons.')

            for i in range(len(self.neurons)):
                activation.append(
                    self.neurons[i].activate(input[i])
                )
        else:
            for i in range(len(self.neurons)):
                activation.append(
                    self.neurons[i].activate()
                )

        return activation

    def get_activations(self):
        return [neuron.get_activation() for neuron in self.neurons]

    def initialize(self):
        for i in range(len(self.neurons)):
            self.neurons[i].initialize()

    def set_trainer(self, trainer):
        for i in range(len(self.neurons)):
            self.neurons[i].set_trainer(trainer)

    def get_connections(self):
        connections = []
        for neuron in self.neurons:
            for connection in neuron.next:
                connections.append(connection)
        return connections

    def set_layer(self, value):
        self.name = value

    def update(self):
        for i in range(len(self.neurons)):
            self.neurons[i].update()

    def to_json(self):
        return {
            'name': self.name,
            'neurons': [neuron.to_json() for neuron in self.neurons]
        }

    def init(self, layer):
        self.set_layer(layer['name'])
        for neuron_obj in layer['neurons']:
            neuron = Neuron(self)
            neuron.init(
                id=neuron_obj['id'],
                activation=neuron_obj['activation'],
                threshold=neuron_obj['threshold'],
                state=neuron_obj['state'],
                old=neuron_obj['old'])
            self.neurons.append(neuron)
