from lib.network import Perceptron, Network
from lib.trainer import Trainer
import json
import os
import unittest

class TestNeuralNetwork(unittest.TestCase):
    def test_XOR_operation(self):
        network = Perceptron(input=2, hidden=[20], output=1)
        trainer = Trainer(network)
        error, epoch = trainer.XOR()
        self.assertTrue(error < 0.05)

        self.assertTrue(abs(sum(network.activate([0, 0])) - 0) < 0.1)
        self.assertTrue(abs(sum(network.activate([0, 1])) - 1) < 0.1)
        self.assertTrue(abs(sum(network.activate([1, 0])) - 1) < 0.1)
        self.assertTrue(abs(sum(network.activate([1, 1])) - 0) < 0.1)

    def test_AND_operation(self):
        network = Perceptron(input=2, hidden=[20], output=1)
        trainer = Trainer(network)
        error, epoch = trainer.AND()
        self.assertTrue(error < 0.05)

        self.assertTrue(abs(sum(network.activate([0, 0])) - 0) < 0.1)
        self.assertTrue(abs(sum(network.activate([0, 1])) - 0) < 0.1)
        self.assertTrue(abs(sum(network.activate([1, 0])) - 0) < 0.1)
        self.assertTrue(abs(sum(network.activate([1, 1])) - 1) < 0.1)

    def test_to_json(self):
        network = Perceptron(input=2, hidden=[2], output=1)
        network_json = network.to_json()
        self.check_network_and_json(network, network_json)

    def test_from_json(self):
        fn = os.path.join(os.path.dirname(__file__), './test_network.json')
        with open(fn) as data_file:
            network_json = json.load(data_file)
            network = Network.from_json(network_json)

            self.check_network_and_json(network, network_json)

    def check_network_and_json(self, network, network_json):
        # Check input layer
        self.assertTrue(len(network.input.neurons) == len(network_json['layers']['input']['neurons']))
        for index in range(len(network.input.neurons)):
            first = network.input.neurons[index]
            second = network_json['layers']['input']['neurons'][index]
            self.assertTrue(first.activation == second['activation'])
            self.assertTrue(first.threshold == second['threshold'])

        # Check hidden layers
        self.assertTrue(len(network.hidden) == len(network_json['layers']['hidden']))
        for index, layer in enumerate(network.hidden):
            self.assertTrue(len(layer.neurons) == len(network_json['layers']['hidden'][index]))
            for index2 in range(len(network.hidden[index].neurons)):
                first = network.hidden[index].neurons[index2]
                second = network_json['layers']['hidden'][index]['neurons'][index2]
                self.assertTrue(first.activation == second['activation'])
                self.assertTrue(first.threshold == second['threshold'])

        # Check output layer
        self.assertTrue(len(network.output.neurons) == len(network_json['layers']['output']['neurons']))
        for index in range(len(network.output.neurons)):
            first = network.output.neurons[index]
            second = network_json['layers']['output']['neurons'][index]
            self.assertTrue(first.activation == second['activation'])
            self.assertTrue(first.threshold == second['threshold'])