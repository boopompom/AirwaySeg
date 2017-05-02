from nn_model import NNModel

num_classes = 32
itr = 2e8

name = "Test2PathNet"
print(name)

NNModel(name=name, num_classes=num_classes).train(
    './Output/{0}_class'.format(num_classes),
    learning_rate=1e-5,
    iterations=itr,
    batch_size=16,
    angles = {
        'train': list(range(0, 360, 10)),
        'validate': list(range(0, 360, 360)),
        'test': list(range(0, 360, 360))
    }
)

