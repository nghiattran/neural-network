from .layer import Layer
import random

def RAN():
    value = random.uniform(-0.5, 0.5)
    return value


class Network(object):
    __id_count__ = 0

    @staticmethod
    def generate_id():
        Network.__id_count__ += 1
        return Network.__id_count__

    def __init__(self, setting = None):
        self.previous = None
        self.next = None
        self.epoch = 0
        self.id = Network.generate_id()

        if setting is None:
            return
        self.set(setting)

    def initialize(self, cb=None):
        if cb is None:
            cb = RAN
        for neuron in self.get_neurons():
            neuron.initialize(cb)

        for conn in self.get_connections():
            conn.initialize(cb)

    def set(self, setting):
        if 'input' not in setting or 'output' not in setting:
            raise ValueError('Input layer or output layer for both are missing')

        if type(setting['input']) is not Layer or type(setting['output']) is not Layer:
            raise ValueError('Input layer, or output layer, or both are not Layer instances')

        self.input = setting['input']
        self.output = setting['output']

        if 'hidden' in setting:
            self.hidden = setting['hidden']
        else:
            self.hidden = []

        self.input.set_name('input')
        self.output.set_name('output')
        for i in range(len(self.hidden)):
            self.hidden[i].set_name(str(i))

        if 'connections' in setting:
            neurons = self.get_neurons_dict()
            for conn in setting['connections']:
                neurons[conn['from']].connect(neurons[conn['to']], conn['weight'])
        else:
            # Initialize all connections and neurons
            self.initialize(setting['init'] if 'init' in setting else None)

        return self

    def get_neurons_dict(self):
        dictionary = {}
        for neuron in self.get_neurons():
            dictionary[neuron.id] = neuron
        return dictionary

    def get_neurons(self):
        neurons = self.input.neurons + self.output.neurons
        for layer in self.hidden:
            neurons += layer.neurons
        return neurons

    def get_connections(self):
        connections = self.input.get_connections() + self.output.get_connections()
        for layer in self.hidden:
            connections += layer.get_connections()
        return connections

    def activate_recursive(self, inputs, round = 0):
        # If previous is not None, this layer is a hidden layer, so, activate the previous layer
        if self.previous is not None:
            # If previous round is equal is this layer round, the previous layer is already executed.
            if self.previous.round == round:
                inputs = self.previous.get_activations()
            else:
                inputs = self.previous.activate_recursive(inputs)
        print(inputs, self.name)
        return self.activate(inputs)

    def activate(self, inputs, epoch = 0):
        self.epoch = epoch
        # If previous is not None, this layer is a hidden layer, so, activate the previous layer
        if self.previous is not None:
            # If previous round is equal is this layer round, the previous layer is already executed.
            if self.previous.epoch == epoch:
                inputs = self.previous.output.get_activations()
            else:
                inputs = self.previous.activate(inputs)

        self.input.activate(inputs)
        for layer in self.hidden:
            layer.activate()
        return self.output.activate()

    def propagate(self, learning_rate, outputs = None, momentum = 0):
        errors = self.output.propagate(learning_rate=learning_rate,
                                       outputs=outputs,
                                       momentum=momentum)
        for i in range(len(self.hidden) - 1, -1, -1):
            self.hidden[i].propagate(learning_rate=learning_rate,
                                     outputs=None,
                                     momentum=momentum)
        self.input.propagate(learning_rate=learning_rate,
                             outputs=None,
                             momentum=momentum)
        return errors

    def set_squash(self, squash):
        for neuron in self.get_neurons():
            neuron.squash = squash

    def project(self, network):
        self.next = network
        network.previous = self
        self.output.project(network.input)
        return self

    def to_json(self):
        return {
            'layers': {
                'input': self.input.to_json(),
                'hidden': [layer.to_json() for layer in self.hidden],
                'output': self.output.to_json()
            },
            'connections': [conn.to_json() for conn in self.get_connections()]
        }

    @staticmethod
    def from_json(setting):
        return Network({
            'input': Layer(setting['layers']['input']),
            'hidden': [Layer(layer_setting) for layer_setting in setting['layers']['hidden']],
            'output': Layer(setting['layers']['output']),
            'connections': setting['connections']
        })


class Perceptron(Network):
    def __init__(self, input, output, hidden = None):
        super(Perceptron, self).__init__()
        input_layer = Layer(input)
        output_layer = Layer(output)

        hidden_layer = []
        if hidden is None:
            input_layer.project(output_layer)
        else:
            for i in range(len(hidden)):
                hidden_layer.append(Layer(hidden[i]))

                if i != 0:
                    hidden_layer[i -1].project(hidden_layer[i])

            input_layer.project(hidden_layer[0])
            hidden_layer[len(hidden_layer) - 1].project(output_layer)

        self.set({
            'input': input_layer,
            'hidden': hidden_layer,
            'output': output_layer
        })