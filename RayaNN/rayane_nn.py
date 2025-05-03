import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import logsumexp

# Classes du modèle

class MLP:
    def __init__(self, din, dout, w=None, b=None):
        self.W = w if w is not None else np.random.rand(dout, din) - 0.5
        self.b = b if b is not None else np.random.rand(dout) - 0.5

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, grad):
        self.deltaW = grad.T @ self.x
        self.deltab = grad.sum(0)
        return grad @ self.W

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        grad[self.x < 0] = 0
        return grad

class LogSoftmax:
    def forward(self, x):
        self.x = x
        return x - logsumexp(x, axis=1)[..., None]

    def backward(self, gradout):
        probs = np.exp(self.x - logsumexp(self.x, axis=1)[..., None])
        batch_size, num_classes = self.x.shape
        grad = np.zeros_like(self.x)
        for b in range(batch_size):
            for i in range(num_classes):
                grad[b, i] = gradout[b, i] - probs[b, i] * gradout[b].sum()
        return grad

class SequentialNN:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

class NLLLoss:
    def forward(self, pred, true):
        self.pred = pred
        self.true = true
        return -np.sum(pred[np.arange(len(true)), true])

    def backward(self):
        grad = np.zeros_like(self.pred)
        grad[np.arange(len(self.true)), self.true] = -1
        return grad

class Optimizer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            if isinstance(layer, MLP):
                layer.W -= self.lr * layer.deltaW
                layer.b -= self.lr * layer.deltab

# Entraînement 

def train(model, optimizer, X_train, Y_train, epochs=5000, batch_size=100):
    loss_fn = NLLLoss()
    for epoch in tqdm(range(epochs)):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        x_batch = X_train[idx]
        y_batch = Y_train[idx]
        pred = model.forward(x_batch)
        loss = loss_fn.forward(pred, y_batch)
        grad = loss_fn.backward()
        model.backward(grad)
        optimizer.step()

if __name__ == "__main__":
    data = pd.read_csv('RayaNN/data_set/train.csv').values
    np.random.shuffle(data)

    X = data[:, 1:] / 255.0
    Y = data[:, 0].astype(int)

    X_train, Y_train = X[1000:], Y[1000:]
    X_dev, Y_dev = X[:1000], Y[:1000]

    model = SequentialNN([
        MLP(784, 10),
        ReLU(),
        MLP(10, 10),
        LogSoftmax()
    ])

    optimizer = Optimizer(model, lr=1e-3)
    train(model, optimizer, X_train, Y_train)

    np.savez("save_model.npz",
             W1=model.layers[0].W, b1=model.layers[0].b,
             W2=model.layers[2].W, b2=model.layers[2].b)

    correct = 0
    for i in range(len(X_dev)):
        pred = model.forward(X_dev[i].reshape(1, -1)).argmax()
        if pred == Y_dev[i]:
            correct += 1
    print("Dev accuracy:", correct / len(X_dev) * 100, "%")
