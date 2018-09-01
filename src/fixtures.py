from network import NeuralNetwork
from layers import *


def network_1(input_num, output_num, lr):
    return NeuralNetwork(
        name="14-100-40-4",
        lr=lr,
        layers=[
            FullyConnectedLayer(input_num, 100),
            ReLU(),
            FullyConnectedLayer(100, 40),
            ReLU(),
            FullyConnectedLayer(40, output_num),
            SoftmaxLayerWithCrossEntropyLoss()
        ]
    )


def network_2(input_num, output_num, lr):
    layers = [
        FullyConnectedLayer(input_num, 28),
        ReLU()
    ]
    for i in range(5):
        layers.append(FullyConnectedLayer(28, 28))
        layers.append(ReLU())
    layers.append(FullyConnectedLayer(28, output_num))
    layers.append(SoftmaxLayerWithCrossEntropyLoss())
    return NeuralNetwork(
        name="14-28X6-4",
        lr=lr,
        layers=layers
    )


def network_3(input_num, output_num, lr):
    layers = [
        FullyConnectedLayer(input_num, 14),
        ReLU()
    ]
    for i in range(27):
        layers.append(FullyConnectedLayer(14, 14))
        layers.append(ReLU())
    layers.append(FullyConnectedLayer(14, output_num))
    layers.append(SoftmaxLayerWithCrossEntropyLoss())
    return NeuralNetwork(
        name="14-14X28-4",
        lr=lr,
        layers=layers
    )
