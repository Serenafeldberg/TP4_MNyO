import matplotlib.pyplot as plt
import numpy as np

def cost_function (A, x, b):
    error = A.dot(x) - b
    return np.dot(error.T, error)

def cost_function_gradient(A, b, x):
    error = np.dot(A, x) - b
    gradient = 2 * np.dot(A.T, error)
    return gradient


def gradient_descent (A, b, s, iterations, x0):
    x = x0

    for _ in range (iterations):
        gradient = cost_function_gradient(A, b, x)
        x = x - s * gradient

    return x

def grad (A, x, b):
    return 2 * (A.T.dot(A).dot(x) - A.T.dot(b))

def learning_rate (A):
    max_val = max (np.linalg.eigvals(A.T.dot(A)))
    return 1 / max_val

if __name__ == "__main__":
    A = np.random.rand(5, 100)
    b = np.random.rand(5, 1)

    print(A)
    print(b)

    s = learning_rate(A)
    x0 = np.random.rand(100, 1)

    x = gradient_descent(A, b, s, 1000, x0)
    print (x)


