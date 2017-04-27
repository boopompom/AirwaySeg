import tensorflow as tf
from nn_model import NNModel

rotation_increment = [360]
repeat_count = 10


for rot in rotation_increment:
    for i in range(repeat_count):
        name = "convnet(aug_5_2x2_fix)_scaled_{0}_{1}".format(str(rot), str(i) )
        print(name)
        itr = 2e8
        NNModel(name=name).train(
            './Output/',
            learning_rate=1e-5,
            iterations=itr,
            batch_size=16,
            angles = {
                'train': list(range(0, 360, rot)),
                'validate': list(range(0, 360, 360)),
                'test': list(range(0, 360, 360))
            }
        )