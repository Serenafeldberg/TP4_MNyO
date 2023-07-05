import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

def cost_function (A, x, b):
    error = A.dot(x) - b
    return np.dot(error.T, error)

def cost_function_gradient(A, b, x):
    error = np.dot(A, x) - b
    gradient = 2 * np.dot(A.T, error)
    return gradient

def gradient_zero (A, b):
    x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
    return x


def regularized_cost_gradient (A, b, x, delta):
    gradient = cost_function_gradient(A, b, x) + 2 * delta * x
    return gradient

def regularizacionL2_gradient (A, b, x, delta):
    gradient = cost_function_gradient(A, b, x) # + derivada de norma 1 de x

def gradient_descent_F (A, b, s, iterations, x0):
    results = []
    x = x0
    for _ in range (iterations):
        gradient = cost_function_gradient(A, b, x)
        print(gradient)
        x = x - s * gradient
        results.append(np.real(cost_function(A, x, b)))
    
    print(np.real(cost_function(A, x, b)).shape)
    return np.array(results), x

def gradient_descent_F2 (A, b, s, delta2, iterations, x0):
    x = x0
    for _ in range (iterations):
        gradient = regularized_cost_gradient(A, b, x, delta2)
        x = x - s * gradient
    return x

def gradient_descent_F1 (A, b, s, delta1, iterations, x0):
    x = x0
    for _ in range (iterations):
        gradient = regularizacionL2_gradient(A, b, x, delta1)
        x = x - s * gradient
    return x

def svd_least_squares(A, b):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_pinv = np.diag(np.reciprocal(S))
    x = Vt.T @ S_pinv @ U.T @ b
    return x


def grad (A, x, b):
    return 2 * (A.T.dot(A).dot(x) - A.T.dot(b))

def learning_rate (A):
    max_val = np.linalg.eigvals(A.T.dot(A)).max()
    return 1 / max_val

def delta2 (A):
    max_eigval = np.linalg.eigvals(A.T.dot(A)).max()
    max_eigval = np.sqrt(max_eigval)
    return (10**(-2))*max_eigval

def delta1 (A):
    max_eigval = np.linalg.eigvals(A.T.dot(A)).max()
    max_eigval = np.sqrt(max_eigval)
    return (10**(-3))*max_eigval

def condicion (A):
    return np.linalg.cond(A, 2)


if __name__ == "__main__":
    """EJERCICIO 1"""
    np.random.seed(49)
    A = np.random.randint(-10, high=10, size=[5,100])
    b = np.random.randint(-10, high=10, size=[5,1])
    #x0 = np.random.randint(0,10,size = [100,1])
    x0 = np.zeros((100, 1))
    s = learning_rate(A)

    x = gradient_zero(A, b)
    # print("Ground truth: ", cost_function(A, x, b))

    results, x = gradient_descent_F(A, b, s, 1000, x0)
    #print ("F:",x[:3])

    #print(results)

    # print("F:", np.real(cost_function(A, x, b)))
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1000), results)
    # plt.xlabel('Iteraciones')
    # plt.ylabel('Costo')
    # plt.title('Evolución del costo con el gradiente descendente')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    delta2 = delta2(A)
    x = gradient_descent_F2(A, b, s, delta2, 10000, x0)
    #print ("F2:",x[:3])

    x = svd_least_squares(A, b)
    #print ("SVD:", x[:3])

    """EJERCICIO 2"""
    A = np.random.rand(100, 100)
    b = np.random.rand(100, 1)

    cond = condicion(A)


