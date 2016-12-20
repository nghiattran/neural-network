from lib.trainer import Trainer as Trainer


class Connection(object):
    def __init__(self, from_node, to_node):
        if type(from_node) is not Neuron or type(to_node) is not Neuron:
            raise ValueError('A neuron must be a "Neuron" instance')
        self.from_node = from_node
        self.to_node = to_node
        self._weight = 0

    def get_weight(self):
        return self._weight

    def set_trainer(self, trainer):
        if type(trainer) is not Trainer:
            raise ValueError('Trainer must be a "Trainer" instance')
        self._trainer = trainer
        return self

    def connection_weight(self):
        return self.from_node.get_activation() * self._weight

    def set_weight(self, weight):
        self._weight = weight
        return self

    def initialize(self):
        self._weight = 0.1

    def get_id(self):
        return '{0}-{1}'.format(self.from_node.get_id(), self.to_node.get_id())

class Neuron(object):
    __id_count__ = 0

    @staticmethod
    def generate_id():
        Neuron.__id_count__ += 1
        return Neuron.__id_count__

    def __init__(self, layer, id = None):
        self.previous = []
        self.next = []
        self._layer = layer
        self.init(
            id = id if id is not None else Neuron.generate_id(),
            activation=0, threshold=0, state=0, old=0
        )

    def init(self, id, activation, threshold, state, old):
        self._id = id
        self.activation = activation
        self.threshold = threshold
        self.state = state
        self.old = old

    def to_json(self):
        return {
            'id': self._id,
            'activation': 0 if self._layer.name == 'input' else self.activation ,
            'threshold': self.threshold,
            'state': self.state,
            'old': self.old,
        }

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
        self._trainer = trainer

        for i in range(len(self.next)):
            self.next[i].set_trainer(trainer)

        return self

    def set_layer(self, layer):
        self._layer = layer
        return self

    def get_activation(self):
        return self.activation

    def get_previous_connections(self):
        return self.previous

    def get_next_connections(self):
        return self.next

    def activate(self, input = None):
        if input is not None:
            self.activation = input
            return

        self.old = self.state
        self.state = 0
        for i in range(len(self.previous)):
            self.state += self.previous[i].connection_weight()

        self.activation = self._trainer.quash(self.state - self.threshold)
        print('activation', self.activation)

    def update(self):
        self._trainer.update(self)

    def propagate(self):
        self._trainer.propagate(self)

    def set_activation(self, activation):
        self.activation = activation
        return self

    def initialize(self):
        for i in range(len(self.next) - 1, -1, -1):
            self.next[i].initialize()

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, value):
        self.threshold = value