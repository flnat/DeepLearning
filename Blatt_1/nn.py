import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    pos_exp = np.exp(z)
    neg_exp = np.exp(z)

    return (pos_exp - neg_exp) / (pos_exp + neg_exp)


def tanh_prime(z):
    return tanh(z) ** 2


def mse(output_activations, y):
    return 1 / 2 * np.mean((y - output_activations) ** 2)


def mse_derivative(output_activations, y):
    return output_activations - y


def logistic_loss(output_activations, y):
    return np.nan_to_num(np.sum(- y * np.log(output_activations) - (1 - y) * np.log(1 - output_activations)))


def logistic_loss_derivative(output_activations, y):
    def inverse_sigmoid(a):
        return np.log(a / (1 - a))

    return np.nan_to_num(np.mean((output_activations - y) * inverse_sigmoid(output_activations)))


class FCNN:

    def __init__(self, **kwargs):
        self.mbs = kwargs.get("mbs", 10)
        self.eta = kwargs.get("eta", 0.03)
        self.no_hidden = kwargs.get("no_hidden", 2)
        self.n_hidden = 4
        self.sizes = kwargs.get("sizes", [2, self.no_hidden, self.no_hidden, self.no_hidden, 1])
        self.num_layers = len(self.sizes)

        # Callables implementing the activation function and the respective derivative
        if kwargs.get("activation_function", "sigmoid") == "sigmoid":
            self.activation_function = sigmoid
            self.activation_function_derivative = sigmoid_prime
        elif kwargs.get("activation_function") == "tanh":
            self.activation_function = tanh
            self.activation_function_derivative = tanh_prime

        # Callables implementing the cost function and the respective derivative
        if kwargs.get("cost_function", "mse") == "mse":
            self.cost_function = mse
            self.cost_function_derivative = mse_derivative
        if kwargs.get("cost_function", "logistic_loss") == "logistic_loss":
            self.cost_function = logistic_loss
            self.cost_function_derivative = logistic_loss_derivative

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.validation_accuracy = None
        self.validation_loss = None

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function(np.dot(w, a) + b)
        return a

    def backprop(self, x, y, ):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]

        zs = []

        # Forward pass
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_function_derivative(activations[-1], y) * self.activation_function_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Output Layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            grad_activation = self.activation_function_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * grad_activation
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def update_mini_batch(self, xmb, ymb):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for i in range(xmb.shape[0]):
            x = np.reshape(xmb[i, :], (xmb.shape[1], 1)).copy()
            if len(ymb.shape) == 2:
                y = np.reshape(ymb[i, :], (ymb.shape[1], 1)).copy()
            else:
                y = ymb[i].copy()

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (self.eta / xmb.shape[0]) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.eta / xmb.shape[0]) * nb for b, nb in zip(self.biases, nabla_b)]

        # return weights, biases

    def evaluate(self, x_eval, y_eval):
        # if labels are one-hot encoded, call evaluate onehot
        if y_eval.shape == 2:
            return self._evaluate_onehot(x_eval, y_eval)
        if self.activation_function.__name__ == "sigmoid":
            return self._evaluate(x_eval, y_eval, 0.5)
        if self.activation_function.__name__ == "tanh":
            return self._evaluate(x_eval, y_eval, 0.0)

    def _evaluate_onehot(self, x_eval, y_eval):
        correct = 0

        for i in range(0, x_eval.shape[0]):
            x = np.reshape(x_eval[i, :], (x_eval[1], 1)).copy()
            if len(y_eval.shape) == 2:
                y = np.reshape(y_eval[i, :], (y_eval[1], 1)).copy()
            else:
                y = y_eval[i].copy()

            y_hat = self.feedforward(x)

            true_class = np.argmax(y_hat)
            predicted_class = np.argmax(y_hat)
            if true_class == predicted_class:
                correct += 1

        return correct

    def _evaluate(self, x_eval, y_eval, cutoff_point):

        y_pred = np.zeros(x_eval.shape[0])
        correct = 0
        for i in range(0, x_eval.shape[0]):
            x = np.reshape(x_eval[i, :], (x_eval.shape[1], 1)).copy()
            y = y_eval[i].copy()

            y_hat = self.feedforward(x)
            y_pred[i] = y_hat
            # Determine squared error for data point

            # Determine if data point was classified correctly
            if y == 0 and y_hat < cutoff_point:
                correct += 1
            if y == 1 and y_hat > cutoff_point:
                correct += 1

        return correct, self.cost_function(y_pred, y_eval)

    def sgd(self, x_train, y_train, x_test, y_test, epochs):

        n_test = x_test.shape[0]
        n_train = x_train.shape[0]

        acc_val, mse_val = np.zeros(epochs), np.zeros(epochs)
        for j in range(epochs):

            p = np.random.permutation(n_train)
            x_train = x_train[p, :]
            y_train = y_train[p]

            for k in range(0, n_train, self.mbs):
                xmb = x_train[k: k + self.mbs, :]
                if len(y_train.shape) == 2:
                    ymb = y_train[k:k + self.mbs, :]
                else:
                    ymb = y_train[k: k + self.mbs]

                self.update_mini_batch(xmb, ymb)

            acc_val[j], mse_val[j] = self.evaluate(x_test, y_test)
            print(f"Epoch {j}: {acc_val[j]} / {n_test} --- {self.cost_function.__name__}: {mse_val[j]}")

        self.validation_accuracy, self.validation_loss = acc_val, mse_val

    def __call__(self, x_train, y_train, x_test, y_test, epochs):
        return self.sgd(x_train, y_train, x_test, y_test, epochs)
