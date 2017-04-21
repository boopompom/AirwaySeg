import random
import numpy as np


class FakeModel:

    def __init__(self, num_classes):
        self.num_classes = num_classes
        pass

    def test(self, input_data):

        num_of_examples = input_data['image'].shape[0]

        results = []
        for i in range(num_of_examples):
            l = []
            for j in range(self.num_classes):
                r = random.randint(0, 100)
                l.append(r)
            l = np.float32(l)
            l /= np.sum(l, axis=0)
            l *= 100
            l = np.int16(l)
            results.append(l)

        return np.array(results)


