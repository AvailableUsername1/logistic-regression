import numpy as np
from sklearn.metrics import accuracy_score

class logisticRegression(object):

    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.random.randn(num_features, 1)
        self.bias = np.random.randn(1, 1)

    def feedforward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        a = sigmoid(z)
        return a

    def GD(self, inputs, target, validation_data, learning_rate = 0.1, iterations = 250):
        print(f"Initial cost {self.evaluate_cost(validation_data)}")
        for i in range(iterations):
            #gradient descent step
            self.a = self.feedforward(inputs)
            self.dz = self.a - target
            self.dw = (1/inputs.shape[0]) * np.dot(np.transpose(inputs), self.dz)
            self.db = np.sum(self.dz) * (1/inputs.shape[0])
            self.weights = self.weights - learning_rate * self.dw
            self.bias = self.bias - learning_rate * self.db

            #printing the cost
            cost = self.evaluate_cost(validation_data)
            print(f"Iteration {i}: cost {cost}")

    def evaluate_cost(self, data):
        """
        Given data (the first column is the label and the rest are features)
        compute the cost with current weights and biases
        """
        #value to add to log function arguments because otherwise it will divide by 0
        epsilon = 1e-7

        x = data[:, 1:]
        y = data [:,[0]]
        a = self.feedforward(x)
        t1 = np.dot(np.transpose(y), np.log(a + epsilon))
        t2 = np.dot(np.transpose(1-y), np.log(1-a + epsilon))
        cost = (-1/data.shape[0]) * (t1 +  t2)
        return cost

    def predict(self, data, threshold = 0.5):
        x = data[:, 1:]
        y = data [:,[0]]
        a = self.feedforward(x)
        a = np.where(a >= threshold, 1.0, a)
        a = np.where(a < threshold, 0, a)
        return a, y

    def accuracy(self, test_data):
        predictions, labels = self.predict(test_data)
        wrong = np.sum(np.square(predictions - labels))
        accuracy =  (1 - wrong/test_data.shape[0]) * 100
        return accuracy



def sigmoid(z):
    return 1 / (1 + np.exp(-z))