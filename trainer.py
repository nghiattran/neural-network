from neuron import Neuron

class Trainer(object):
    def __init__(self):
        self.threshold = 0.2
        self.learning_rate = 0.1

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def calculate_output(self, neuron):
        return self.activation_function(neuron.calculate_connections_sum() - self.threshold)