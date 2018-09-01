from layers import *
from utils import *
import time


class NeuralNetwork:
    def __init__(self, name, lr, layers):
        self.name = name
        self.lr = lr
        self.layers = layers

    def assign(self, weights, biases):
        idx = 0
        for layer in self.layers:
            if type(layer) is FullyConnectedLayer:
                layer.W = weights[idx]
                layer.b = biases[idx]
                idx += 1
        return self

    def forward_propagation(self, X, y=None):
        for layer in self.layers:
            if type(layer) is not SoftmaxLayerWithCrossEntropyLoss:
                X = layer.forward(X)
            else:
                return layer.eval(X, y)

    def backward_propagation(self):
        for layer in reversed(self.layers):
            if type(layer) is SoftmaxLayerWithCrossEntropyLoss:
                grad = layer.backward()
            else:
                grad = layer.backward(grad)

    def gradient_descent(self, lr=0.01, decay=1.0):
        for layer in reversed(self.layers):
            if type(layer) is FullyConnectedLayer:
                lr *= decay
                layer.W -= lr * layer.gW
                layer.b -= lr * layer.gb

    def train_batch(self, X, y):
        batch_loss, batch_acc = self.forward_propagation(X, y)
        self.backward_propagation()
        self.gradient_descent()
        return batch_loss, batch_acc

    def train(self, X_train, y_train, X_val, y_val, iterations, batch_size, save_plot_to_directory):
        start_time = time.time()
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        for itr in range(iterations):
            shuffled_X, shuffled_y = unison_shuffled_copies(X_train, y_train)
            itr_loss = []
            itr_accuracy = []
            for i in range(0, len(shuffled_X), batch_size):
                batch_loss, batch_accuracy = self.train_batch(shuffled_X[i:i+batch_size], shuffled_y[i:i+batch_size])
                itr_loss.append(batch_loss)
                itr_accuracy.append(batch_accuracy)

            # average loss and accuracy of the batch
            train_loss.append(np.sum(itr_loss, axis=0) / len(itr_loss))
            train_acc.append(np.sum(itr_accuracy, axis=0) / len(itr_accuracy))

            # validation loss and accuracy
            loss, acc = self.test(X_val, y_val)
            val_loss.append(loss)
            val_acc.append(acc)

        elapsed_time = time.time() - start_time
        print("Time taken: {0:.2f}s".format(elapsed_time))
        if save_plot_to_directory:
            plot_epoch(save_plot_to_directory,
                       self.name,
                       self.lr,
                       batch_size,
                       iterations,
                       train_loss,
                       train_acc,
                       val_loss,
                       val_acc,
                       elapsed_time)

    def test(self, X_test, y_test):
        loss, acc = self.forward_propagation(X_test, y_test)
        return loss, acc

    def export_gradients(self, directory):
        with open(f"{directory}/dw-{self.name}.csv", "w+") as csv_file:
            for layer in self.layers:
                if type(layer) is FullyConnectedLayer:
                    np.savetxt(csv_file, layer.gW, fmt="%.6f", delimiter=",")
        with open(f"{directory}/db-{self.name}.csv", "w+") as csv_file:
            for layer in self.layers:
                if type(layer) is FullyConnectedLayer:
                    np.savetxt(csv_file, layer.gb, fmt="%.6f", delimiter=",")



