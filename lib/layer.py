from lib.neuron import Neuron


class Layer(object):
    def __init__(self, number):
        if type(number) is int:
            self.neurons = []
            for i in range(number):
                self.neurons.append(Neuron())
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

    def train_weight(self):
        for i in range(len(self.neurons) - 1, -1, -1):
            self.neurons[i].train_weight()

    def activate(self, input = None):
        activation = []

        if input is not None:
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
