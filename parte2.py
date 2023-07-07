import numpy as np
import matplotlib.pyplot as plt
import random

m = 5  # Número de filas de A
n = 100  # Número de columnas de A

# Generar matrices A y vector b aleatorios
A = np.random.rand(m, n)
b = np.random.rand(m, 1)

def cost_function(A, x, b):
    error = A.dot(x) - b
    return np.dot(error.T, error)

def cost_function_gradient(A, b, x):
    error = np.dot(A, x) - b
    gradient = 2 * np.dot(A.T, error)
    return gradient

def regularized_cost_function(A, x, b, delta):
    error = A.dot(x) - b
    return np.dot(error.T, error) + delta * np.linalg.norm(x)**2

def regularized_cost_function_gradient(A, b, x, delta):
    error = np.dot(A, x) - b
    gradient = 2 * np.dot(A.T, error) + 2 * delta * x
    return gradient

def regu2_cost_function(A, x, b, delta):
    error = A.dot(x) - b
    return np.dot(error.T, error) + delta * np.linalg.norm(x, ord=1)

def regu2_cost_function_gradient(A, b, x, delta):
    return 2 * (np.dot(A.T ,(A.dot(x))) - np.dot(A.T,b))  + delta *  np.sign(x)

def gradient_descent(A, b, s, iterations, x0):
    results = []
    gradients = []
    x = x0
    for _ in range(iterations):
        gradient = cost_function_gradient(A, b, x)
        x = x - s * gradient
        results.append(np.real(cost_function(A, x, b)))
        gradients.append(np.linalg.norm(gradient, 2))
    return x, np.array(results), np.array(gradients)

def regularized_gradient_descent(A, b, s, delta, iterations, x0):
    x = x0
    results = []
    gradients = []
    for _ in range(iterations):
        gradient = regularized_cost_function_gradient(A, b, x, delta)
        x = x - s * gradient
        results.append(np.real(regularized_cost_function(A, x, b, delta)))
        gradients.append(np.linalg.norm(gradient, 2))
    return x, np.array(results), np.array(gradients)

def regu2_gradient_descent(A, b, s, delta, iterations, x0):
    x = x0
    results = []
    gradients = []
    for _ in range(iterations):
        gradient = regu2_cost_function_gradient(A, b, x, delta)
        x = x - s * gradient
        results.append(np.real(regu2_cost_function(A, x, b, delta)))
        gradients.append(np.linalg.norm(gradient, 2))
    return x, np.array(results), np.array(gradients)

def learning_rate(A):
    max_val = np.linalg.eigvals(2*A.T.dot(A)).max()
    return 1 / max_val

def delta2(A):
    max_eigval = np.linalg.eigvals(2*A.T.dot(A)).max()
    return (10**(-2)) * max_eigval

def delta1(A):
    max_eigval = np.linalg.eigvals(2*A.T.dot(A)).max()
    return (10**(-3)) * max_eigval

s = learning_rate(A)
delta2_value = delta2(A)
delta1_value = delta1(A)
x0 = np.random.rand(n, 1)

x_svd = np.linalg.pinv(A) @ b  # Solución mediante SVD

x_f, results1, gradients_F = gradient_descent(A, b, s, 1000, x0)  # Solución minimizando F
x_f2, results2, gradients_F2 = regularized_gradient_descent(A, b, s, delta2_value, 1000, x0)  # Solución minimizando F2
x_f1, results3, gradients_F1 = regu2_gradient_descent(A, b, s, delta1_value, 1000, x0)  # Solución minimizando F1

# Comparar las soluciones obtenidas con la solución mediante SVD
cost_svd = cost_function(A, x_svd, b)
cost_f = cost_function(A, x_f, b)
cost_f2 = regularized_cost_function(A, x_f2, b, delta2_value)
cost_f1 = regu2_cost_function(A, x_f1, b, delta1_value)




"""EJERCICIO 2"""

def plot_error (errors, iterations, title):
    plt.plot(iterations, errors)
    plt.xlabel('Iteraciones')
    plt.ylabel('Error')
    plt.title(title)
    plt.show()

def generate_random_matrix_with_condition(m, n, condition_number):
    # Generar una matriz aleatoria A de tamaño m x n
    A = np.random.rand(m, n)

    # Calcular la descomposición SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    S[0] = condition_number **2
    S[-1] = condition_number

    for i in range(1, len(S)-1):
        S[i] = random.uniform(S[-1], S[0])

    # Reconstruir la matriz A con los valores singulares modificados
    A_modified = U @ np.diag(S) @ Vt

    return A_modified

def condition_number(matrix):

    _, singular_values, _ = np.linalg.svd(matrix)
    condition_number = np.max(singular_values) / np.min(singular_values)
    return condition_number

def gradient_descent_iterations(A, b, tol, s):
    n = A.shape[1]  # Número de columnas de A

    errors = []

    x = np.random.rand(n, 1)  # Condición inicial aleatoria
    error = np.real(cost_function(A, x, b))[0][0]
    iterations = 0

    while error > tol:
        gradient = cost_function_gradient(A, b, x)
        x = x - (s/n) * gradient
        error = np.real(cost_function(A, x, b))[0][0]
        errors.append(error)
        iterations += 1

    return iterations, errors

def K (A):
    max_eigval = np.linalg.eigvals(A.T.dot(A)).max()
    norm_A = np.sqrt(max_eigval)
    max_eigval_inv = np.linalg.eigvals(np.linalg.inv(A.T.dot(A))).max()
    norm_A_inv = np.sqrt(max_eigval_inv)
    return norm_A * norm_A_inv

tol = 1e-2
n = 100
condition_num = 1
A = generate_random_matrix_with_condition(n, n, condition_num)
b = np.random.rand(n, 1)  # Vector b aleatorio
s = learning_rate(A)

condition_numbers = [1.1, 3, 10]  # Números de condición de A

fig, axes = plt.subplots(1, 3)
i = 0

# Calcular número de iteraciones y comparar con predicción teórica
for cond_number in condition_numbers:
    A = generate_random_matrix_with_condition(m, n, cond_number)
    print(f"CONDICIONADA? {K(A)}")
    print(f"condicion: {condition_number(A)}")
    s = learning_rate(A)

    b = np.random.rand(m, 1)  # Vector b aleatorio

    predicted_iterations = np.log(1 / tol) / np.log(cond_number)
    actual_iterations, errors = gradient_descent_iterations(A, b, tol, s)

    axes[i].plot(np.arange(1, actual_iterations + 1), np.array(errors))
    axes[i].set_xlabel('Iteraciones')
    axes[i].set_ylabel('Error')
    axes[i].set_title(f"Error para condición {cond_number}")


    #plot_error(np.array(errors), np.arange (1, actual_iterations + 1), f"Error para condición {cond_number}")

    print(f"Número de condición de A: {np.linalg.cond(A)}")
    print(f"Número de iteraciones (predicción teórica): {predicted_iterations}")
    print(f"Número de iteraciones (obtenido): {actual_iterations}")
    print("---------")
    i+=1

plt.subplots_adjust(wspace=0.3)
# Mostrar gráfico
plt.show()


