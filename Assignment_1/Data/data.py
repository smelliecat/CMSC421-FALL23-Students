# Don't MODIFY this code block!
class Data:
    """
    Stores an input array of training data, and hands it to the next layer.
    THIS AND THE INPUT LAYER COULD ESSENTIALLY PERFORM THE SAME FUCTION. 
    THEY HAVE BEEN SEPERATED SOLELY FOR LEARNING PURPOSES!!
    """

    def __init__(self, data):
        self.data = data
        self.data = self.data.reshape(-1, 1)
        self.output_dimension = self.data.shape
        print("in DATA class", self.output_dimension)

    def set_data(self, data):
        self.data = data

    def forward(self):
        return self.data

    def backward(self, dwnstrm):
        pass


