import sys
import numpy as np
from buildnet1 import NeuralNetwork


def load_weights(file_name):
    with open(file_name, "r") as file:
        # Read the first line and get the size of the network
        layer1_size, layer2_size = map(int, file.readline().split())

        # Read the weights for the first layer
        weights1 = [[float(x) for x in file.readline().split()] for _ in range(layer1_size)]

        # Read the weights for the second layer
        weights2 = [[float(x) for x in file.readline().split()] for _ in range(layer2_size)]

    return weights1, weights2


if __name__ == '__main__':
    weights_file = "wnet1.txt"
    test_file = "testnet1.txt"
    # Load weights from file
    weights1, weights2 = load_weights(weights_file)

    # Create the neural network with loaded weights
    neural_network = NeuralNetwork(weights1, weights2)

    # Load data from file
    with open(test_file, 'r') as file:
        data = file.read().splitlines()

    # Make predictions using the neural network
    predictions = []
    for data_point in data:
        # input_data = [int(bit) for bit in data_point]
        input_data = np.array([int(bit) for bit in data_point])
        prediction = neural_network.forward(input_data)
        # print(prediction)
        predictions.append(int(round(prediction[0])))

    # Write predictions to output file
    with open("labels1.txt", 'w') as file:
        for prediction in predictions:
            file.write(str(prediction) + "\n")
