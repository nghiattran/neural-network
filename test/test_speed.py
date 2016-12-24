from lib.network import Perceptron, Network
from lib.trainer import Trainer, Cost
import json
import os
import unittest
import time

class TestSpeed(unittest.TestCase):
    def test_from_json(self):
        return
        fn = os.path.join(os.path.dirname(__file__), './test_network.json')
        settings = {
            'momentum': 0.99,
            'epoch': 5000,
            'log': 100,
            'error': 0.1,
            'rate': 0.2,
            'cost': Cost.SE
        }
        with open(fn) as data_file:
            network_json = json.load(data_file)
            network = Network.from_json(network_json)
            trainer = Trainer(network)
            start = time.time()
            error, epoch = trainer.XOR(settings)
            stop = time.time()
            print(stop - start)