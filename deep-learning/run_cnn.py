from cnn import Cnn

if __name__ == '__main__':
    iterations, hidden_layer, learning_rate, dropout_probability = 1000, 50, 0.0001, 0.5

    cnn = Cnn(iterations, hidden_layer, learning_rate, dropout_probability)
    cnn.train()
