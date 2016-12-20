from trainer import Trainer

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

    def calculate_output(self):
        if self.trainer is None:
            raise ValueError('Trainer must be set before trainning')
        return self.trainer.calculate_output(self)

    def calculate_input(self):
        return self.from_node.get_weight() * self.weight


class Neuron(object):
    __id_count__ = 0

    @staticmethod
    def generate_id():
        Neuron.__id_count__ += 1
        return Neuron.__id_count__

    def __init__(self):
        self._id = Neuron.generate_id()
        self.error = 0
        self.weight = 0
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
        return self

    def set_layer(self, layer):
        self.layer = layer
        return self

    def get_weight(self):
        return self.weight

    def calculate_connections_sum(self):
        sum = 0
        for i in range(len(self.previous)):
            sum += self.previous.calculate_input()
        return sum

    def calculate_output(self):
        if self.trainer is None:
            raise ValueError('Trainer must be set before trainning')
        return self.trainer.calculate_output(self)