def Perceptron(object):
    def set(self, setting):
        try:
            self.input = setting['input']
            self.output = setting['output']
        except:
            raise Exception('Input layer or output layer for both are missing')

        if 'hidden' in setting:
            self.hidden = setting['hidden']
        else:
            self.hiddent = None