import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rayane_nn import *

def main():
    data = pd.read_csv("/Users/rayanedakhlaoui/Desktop/Kaggle/Digit Recognizer/RayaNN/data_set/test.csv").values
    idx = np.random.randint(0, len(data))
    X = data[idx] / 255.0
    image = X.reshape(28, 28)

    weights = np.load("save_model.npz")
    model = SequentialNN([
        MLP(784, 10, weights['W1'], weights['b1']),
        ReLU(),
        MLP(10, 10, weights['W2'], weights['b2']),
        LogSoftmax()
    ])

    prediction = model.forward(X.reshape(1, -1)).argmax()

    plt.imshow(image, cmap="gray")
    plt.title(f"Prediction: {prediction} (Image #{idx})")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()