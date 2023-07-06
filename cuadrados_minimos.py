import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
        x = x - s * gradient
        results.append(np.real(cost_function(A, x, b)))
    
    return np.array(results), x

def gradient_descent_F2 (A, b, s, delta2, iterations, x0):
    x = x0
    results = []
    for _ in range (iterations):
        gradient = regularized_cost_gradient(A, b, x, delta2)
        x = x - s * gradient
        results.append(np.real(cost_function(A, x, b)))
    return np.array(results), x

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

def plot (iterations, results, x):
    # Gráfico de la evolución de la solución
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), results.flatten())  # Aplanar el array cost_history
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    plt.axhline(y=0, color='r', alpha=0.7, linestyle='dashed')
    plt.title('Evolución del costo con el gradiente descendente')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Crear el gráfico de calor con la parte real de la solución
    plt.figure(figsize=(10, 6))
    sns.heatmap(np.real(x.reshape((10, 10))), cmap='coolwarm', cbar=True)
    plt.title('Heatmap de la parte real de la solución del gradiente descendente')
    plt.xlabel('Columnas')
    plt.ylabel('Filas')
    plt.show()


if __name__ == "__main__":
    """EJERCICIO 1"""
    iterations = 1000
    np.random.seed(49)
    A = np.random.randint(-10, high=10, size=[5,100])
    b = np.random.randint(-10, high=10, size=[5,1])
    #x0 = np.random.randint(0,10,size = [100,1])
    x0 = np.zeros((100, 1))
    s = learning_rate(A)

    x = gradient_zero(A, b)
    print("Ground truth: ", cost_function(A, x, b))

    x = svd_least_squares(A, b)
    print ("SVD:", cost_function(A, x, b))

    results, x = gradient_descent_F(A, b, s, iterations, x0)
    print("RESULT:", cost_function(A, x, b))

    plot(iterations, results, x)


    delta2 = delta2(A)
    results, x = gradient_descent_F2(A, b, s, delta2, iterations, x0)
    
    plot(iterations, results, x)

    """EJERCICIO 2"""
    A = np.random.rand(100, 100)
    b = np.random.rand(100, 1)

    cond = condicion(A)


