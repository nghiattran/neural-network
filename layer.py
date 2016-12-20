from neuron import Neuron

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