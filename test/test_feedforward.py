import unittest
from neuralnetwork.ann import ANN

class TestFeedforward(unittest.TestCase):

    def test_feedforward(self):
        ann = ANN()
        ann.add_layer(10)