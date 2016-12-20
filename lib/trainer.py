# # import lib.architect as architect
# from .architect import Perceptron
import math


class Trainer(object):
    def __init__(self, neural_network = None):
        if neural_network is not None:
            self.set_network(neural_network)

        self.threshold = 0.2
        self.learning_rate = 0.1

    def set_network(self, neural_network):
        # if neural_network is not None and type(neural_network) is not Perceptron:
        #     raise ValueError('Neural Network object is not a Perceptron instance')
        self.neural_network = neural_network
        self.neural_network.set_trainer(self)

    def quash(self, x):
        # return 1 if x >= 0 else 0
        return 1 / (1 + pow(math.e, -1 * x))

    def propagate(self, neuron):
        activation = neuron.get_activation()
        if neuron.layer.name == 'output':
            neuron.error_gradient = activation * (1 - activation) * self.error
        else:
            sum = 0
            for i in range(len(neuron.next)):
                to_node = neuron.next[i].to_node
                sum += to_node.error_gradient * neuron.next[i].get_weight()
            neuron.error_gradient = activation * (1 - activation) * sum

        for connection in neuron.previous:
            connection.delta = self.learning_rate * connection.from_node.activation * neuron.error_gradient

        neuron.delta_threshold = self.learning_rate * -1 * neuron.error_gradient

    def update(self, neuron):
        neuron.threshold + neuron.delta_threshold
        for connection in neuron.previous:
            connection.set_weight(connection.get_weight() + connection.delta)
            print(connection.get_id(), connection.get_weight(), connection.delta)
        print(neuron.get_id(), neuron.threshold, neuron.delta_threshold)

    def train(self, training_set, setting = None):
        connections = self.neural_network.get_connections()
        values = [0.5, 0.9, 0.4, 1, -1.2, 1.1]
        for i in range(len(connections)):
            connections[i].set_weight(values[i])

        neurons = self.neural_network.get_neurons()
        values = [0, 0, 0.8, -0.1, 0.3]
        for i in range(len(neurons)):
            neurons[i].set_threshold(values[i])

        self.neural_network.activate([1, 1])
        outputs = self.neural_network.get_outputs()
        print(outputs)
        self.error = 0
        for index in range(len(outputs)):
            self.error += 0 - outputs[index]
        # print(self.error)
        self.neural_network.propagate()

        # Initialization
        # self.neural_network.initialize()
        # connection = [neuron.next for neuron in self.neural_network.input.get_neurons()]
        # connections = [connection[0][0], connection[1][0]]
        # connection[0][0].set_weight(0.3)
        # connection[1][0].set_weight(-0.1)
        #
        # self.error = 0
        # error = 999
        # count = 0
        # while count < 3:
        #     count += 1
        #     error = 0
        #     for i in range(len(training_set)):
        #         # Activation
        #         self.neural_network.activate(training_set[i]['input'])
        #
        #         # Calculate error
        #         outputs = self.neural_network.get_outputs()
        #         if 'output' not in training_set[i] and len(outputs) != len(training_set[i]['output']):
        #             raise ValueError("Number of ouputs and number of desired outputs don't macth")
        #         self.error = 0
        #         for index in range(len(outputs)):
        #             self.error += pow(training_set[i]['output'][index] - outputs[index], 2)
        #
        #         # Train weight
        #         self.neural_network.train_weight()
        #         # print(self.error)
        #         error += self.error
        #
        #         print('output: {0}, final weights: {1}, {2}, error: {3}'.format(
        #             outputs, connections[0].get_weight(), connections[1].get_weight(),
        #             self.error
        #         ))

    def XOR(self):
        self.train([
            {
                'input': [1, 1],
                'output': [0]
            }

        ])

