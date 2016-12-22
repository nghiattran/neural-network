from lib.network import Network
from lib.layer import Layer
from lib.trainer import Trainer
import json


def update_rate(rate, error, last_error):
    if error > last_error:
        return rate * 0.85
    else:
        return rate * 1.05;


with open('data.txt') as data_file:
    # data = json.load(data_file)
    # new = Network.from_json(data)
    # print(new.to_json())


    input_layer = Layer(2)
    hidden_layer = Layer(20)
    output_layer = Layer(1)

    input_layer.project(hidden_layer)
    hidden_layer.project(output_layer)

    new = Network({
        'input': input_layer,
        'hidden': [hidden_layer],
        'output': output_layer
    })

    # print(new.to_json())

    trainer = Trainer(new)
    settings = {
        'shuffle': True,
        'momentum': 0.99,
        'error': 0.005,
        'epoch': 5000,
        # 'log': 1,
        # 'update': update_rate
    }

    error, epoch = trainer.XOR()

    print(error, epoch)

    # with open('res.json', 'w') as file1:
    #     json.dump(new.to_json(), file1, sort_keys=True, indent=4, ensure_ascii=False)

    print(new.activate([0, 0]))
    print(new.activate([0, 1]))
    print(new.activate([1, 0]))
    print(new.activate([1, 1]))