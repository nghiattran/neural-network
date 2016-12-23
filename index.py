from lib.layer import Layer
from lib.network import Network, Perceptron
from lib.trainer import Trainer
import queue, os, json

errors = {
    'last_error': 0,
    'sign': 1,
    'count': 0
}
pivot = 5


def adapt(learning_rate, error = 0, last_error = 0):
    if learning_rate == -1:
        return 0.1

    if error - last_error >= 0 and errors['sign'] >= 0:
        errors['count'] += 1
        if errors['count'] - 1 > pivot:
            return learning_rate * 1.05
    elif error - last_error < 0 and errors['sign'] < 0:
        errors['count'] += 1
        if errors['count'] - 1 > pivot:
            return learning_rate * 0.85
    else:
        errors['count'] = 0
        errors['sign'] = error - last_error
    return learning_rate

network = Perceptron(input=2, hidden=[2], output=1)
network1 = Perceptron(input=1, hidden=[2], output=1)
network.project(network1)
# trainer = Trainer(network)
# settings = {
#     'shuffle': True,
#     'momentum': 0.99,
#     'epoch': 500,
#     'log': 5,
#     'error': 0.01,
#     'rate': adapt
# }
# error, epoch = trainer.AND(settings)

print(network1.activate([0, 0]))
print(network1.activate([1, 0]))
print(network1.activate([0, 1]))
print(network1.activate([1, 1]))



# fn = os.path.join(os.path.dirname(__file__), './test/test_network.json')
# with open(fn) as data_file:
#     network_json = json.load(data_file)
#     network = Network.from_json(network_json)
#     trainer = Trainer(network)
#     trainer.AND({
#         'log': 1,
#         'epoch': 5
#     })