# # import lib.architect as architect
# from .architect import Perceptron

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

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def activate(self, input):
        return self.activation_function(input - self.threshold)

    def calculate_weight_correction(self, connection):
        # print(connection.from_node.get_activation(), self.learning_rate, self.error)
        return connection.from_node.get_activation() * self.learning_rate * self.error

    def train_weight(self, connection):
        connection.set_weight(
            connection.get_weight() + self.calculate_weight_correction(connection)
        )

    def train(self, training_set, setting = None):
        connections = self.neural_network.get_connections()
        values = [0.5, 0.9, 0.4, 1, -1.2, 1.1]
        for conn in connections:

            print(conn.get_id())


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
                'input': [0, 0],
                'output': [0]
            },{
                'input': [0, 1],
                'output': [0]
            },{
                'input': [1, 0],
                'output': [0]
            },{
                'input': [1, 1],
                'output': [1]
            }

        ])

