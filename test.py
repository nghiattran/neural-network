from layer import Layer
from architect import Perceptron

input_layer = Layer(5)
hiddent_layer = Layer(5)
output_layer = Layer(5)

input_layer.project(hiddent_layer)
hiddent_layer.project(output_layer)

perceptron = Perceptron({
    'input': input_layer,
    'hiddent': hiddent_layer,
    'output': output_layer
})