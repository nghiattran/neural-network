from lib.trainer import Trainer as Trainer


class Connection(object):
    def __init__(self, from_node, to_node):
        if type(from_node) is not Neuron or type(to_node) is not Neuron:
            raise ValueError('A neuron must be a "Neuron" instance')
        self.from_node = from_node
        self.to_node = to_node
        self.weight = 0

    def get_weight(self):
        return self.weight

    def set_trainer(self, trainer):
        if type(trainer) is not Trainer:
            raise ValueError('Trainer must be a "Trainer" instance')
        self.trainer = trainer
        return self

    def train_weight(self):
        self.trainer.train_weight(self)

    def set_weight(self, weight):
        self.weight = weight
        return self

    def initialize(self):
        self.weight = 0.1

    def get_id(self):
        return '{0}-{1}'.format(self.from_node.get_id(), self.to_node.get_id())

class Neuron(object):
    __id_count__ = 0

    @staticmethod
    def generate_id():
        Neuron.__id_count__ += 1
        return Neuron.__id_count__

    def __init__(self):
        self._id = Neuron.generate_id()
        self.error = 0
        self.activation = 0
        self.previous = []
        self.next = []

    def get_id(self):
        return self._id

    def connect(self, next_neuron):
        if type(next_neuron) is not Neuron:
            raise ValueError('A neuron must be a "Neuron" instance')
        connection = Connection(from_node=self, to_node=next_neuron)
        self.set_next(connection)
        next_neuron.set_previous(connection)
        return self

    def set_previous(self, connection):
        if type(connection) is not Connection:
            raise ValueError('A connection must be a "Connection" instance')
        self.previous.append(connection)
        return self

    def set_next(self, connection):
        if type(connection) is not Connection:
            raise ValueError('A connection must be a "Connection" instance')
        self.next.append(connection)
        return self

    def set_trainer(self, trainer):
        if type(trainer) is not Trainer:
            raise ValueError('Trainer must be a "Trainer" instance')
        self.trainer = trainer

        for i in range(len(self.next)):
            self.next[i].set_trainer(trainer)

        return self

    def set_layer(self, layer):
        self.layer = layer
        return self

    def get_activation(self):
        return self.activation

    def get_previous_connections(self):
        return self.previous

    def get_next_connections(self):
        return self.next

    def calculate_input(self, input = None):
        if input is None:
            input = self.activation

        sum = 0
        for i in range(len(self.previous)):
            sum += self.previous[i].get_weight() * input
        return sum

    def activate(self, input = None):
        self.activation = self.trainer.activate(self.calculate_input(input))

    def train_weight(self):
        for i in range(len(self.previous) - 1, -1, -1):
            self.previous[i].train_weight()

    def set_activation(self, activation):
        self.activation = activation
        return self

    def initialize(self):
        for i in range(len(self.next) - 1, -1, -1):
            self.next[i].initialize()