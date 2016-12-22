from lib.neuron import Neuron


class Layer(object):
    def __init__(self, setting):
        self.neurons = []

        if type(setting) is int:
            self.name = ''
            for i in range(setting):
                self.neurons.append(Neuron(self))
        elif type(setting) is dict:
            try:
                self.name = setting['name']
                for neuron in setting['neurons']:
                    self.neurons.append(Neuron(self, neuron))
            except:
                raise ValueError('Input file is corrupted.')
        else:
            raise ValueError('Layer constructor only takes either an integer argument for a dictionary.')

    def to_json(self):
        return {
            'name': self.name,
            'neurons': [neuron.to_json() for neuron in self.neurons]
        }

    def set_name(self, name):
        self.name = name

    @staticmethod
    def from_json(setting):
        return Layer(setting)

    def activate(self, inputs = None):
        if inputs is None:
            return [self.neurons[i].activate() for i in range(len(self.neurons))]

        if len(inputs) != len(self.neurons):
            raise ValueError('Input size does not match number of neurons.')

        return [self.neurons[i].activate(inputs[i]) for i in range(len(self.neurons))]

    def propagate(self, outputs = None):
        if outputs is None:
            return [self.neurons[i].propagate() for i in range(len(self.neurons)) ]

        if len(outputs) != len(self.neurons):
            raise ValueError('Output size does not match number of neurons.')

        return [self.neurons[i].propagate(outputs[i]) for i in range(len(self.neurons))]

    def project(self, layer):
        if type(layer) is not Layer:
            raise ValueError('Projected object is not a Layer instance')

        for neuron in self.neurons:
            for projected_neuron in layer.neurons:
                neuron.connect(projected_neuron)

    def get_connections(self):
        connections = []
        for neuron in self.neurons:
            connections += neuron.next
        return connections

    def set_trainer(self, trainer):
        for neuron in self.neurons:
            neuron.set_trainer(trainer)