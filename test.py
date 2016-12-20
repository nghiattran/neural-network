from lib.architect import Perceptron
from lib.layer import Layer
from lib.trainer import Trainer

input_layer = Layer(2)
hidden_layer = Layer(2)
output_layer = Layer(1)

input_layer.project(hidden_layer)
hidden_layer.project(output_layer)

perceptron = Perceptron({
    'input': input_layer,
    'hidden': [hidden_layer],
    'output': output_layer
})

trainer = Trainer(perceptron)
trainer.XOR()
# print('test')
# print(perceptron.activate([0, 0]))
# print(perceptron.activate([0, 1]))
# print(perceptron.activate([1, 0]))
# print(perceptron.activate([1, 1]))

print(perceptron.to_json())

perceptron1 = Perceptron.from_json(perceptron.to_json())

print(perceptron1.to_json())