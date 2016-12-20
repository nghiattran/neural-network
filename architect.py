from layer import Layer

class Perceptron(object):
    def __init__(self, setting):
        if 'input' not in setting or 'output' not in setting:
            raise Exception('Input layer or output layer for both are missing')

        if type(setting['input']) is not Layer or type(setting['output']) is not Layer:
            raise ValueError('Input layer, or output layer, or both are not Layer instances')

        self.input = setting['input']
        self.output = setting['output']

        if 'hidden' in setting:
            self.hidden = setting['hidden']
        else:
            self.hiddent = None

    def get_layers(self):
        layers =  [self.input, self.output]
        layers[1:1] = self.hidden
        return layers