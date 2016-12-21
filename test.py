# from lib.architect import Perceptron
# from lib.layer import Layer
# from lib.trainer import Trainer
#
# input_layer = Layer(2)
# hidden_layer = Layer(2)
# output_layer = Layer(1)
#
# input_layer.project(hidden_layer)
# hidden_layer.project(output_layer)
#
# perceptron = Perceptron({
#     'input': input_layer,
#     'hidden': [hidden_layer],
#     'output': output_layer
# })
#
# trainer = Trainer(perceptron)
# trainer.XOR()
# # print('test')
# # print(perceptron.activate([0, 0]))
# # print(perceptron.activate([0, 1]))
# # print(perceptron.activate([1, 0]))
# # print(perceptron.activate([1, 1]))
#
# print(perceptron.to_json())
#
# perceptron1 = Perceptron.from_json(perceptron.to_json())
#
# print(perceptron1.to_json())

from lib.network import Network
from lib.layer import Layer
from lib.trainer import Trainer
import json

# input_layer = Layer(2)
# hidden_layer = Layer(2)
# output_layer = Layer(1)
#
# input_layer.project(hidden_layer)
# hidden_layer.project(output_layer)
#
# perceptron = Network({
#     'input': input_layer,
#     'hidden': [hidden_layer],
#     'output': output_layer
# })
#
# data = perceptron.to_json()
# print(data)
# with open('data.txt', 'w') as outfile:
#     json.dump(data, outfile, sort_keys = True, indent = 4, ensure_ascii=False)

with open('data.txt') as data_file:
    data = json.load(data_file)
    new = Network.from_json(data)
    # print(new.to_json())
    trainer = Trainer(new)
    trainer.train([{
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
    }])

    print(new.activate([0, 0]))
    print(new.activate([0, 1]))
    print(new.activate([1, 0]))
    print(new.activate([1, 1]))