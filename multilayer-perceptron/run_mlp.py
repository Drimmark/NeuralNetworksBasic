from mlp import Mlp

if __name__ == '__main__':
    iterations, hidden_layer, learning_rate = 1000, 50, 0.1

    mlp = Mlp(iterations, hidden_layer, learning_rate)
    mlp.train()
