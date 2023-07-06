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

print("Solución mediante SVD:")
print(x_svd[:3])
print("Costo (F):", cost_svd)
print()
print("Solución minimizando F:")
print(x_f[:3])
print("Costo (F):", cost_f)
print()
print("Solución minimizando F2:")
print(x_f2[:3])
print("Costo (F2):", cost_f2)
print()
print("Solución minimizando F1:")
print(x_f1[:3])
print("Costo (F1):", cost_f1)

# Resultados de las iteraciones
iterations = np.arange(1, 1001)

# Crear el gráfico de líneas
plt.plot(iterations, results1.flatten(), label='F')
plt.plot(iterations, results2.flatten(), label='F2')
plt.plot(iterations, results3.flatten(), label='F1')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.ylim(bottom=-1, top=10)
plt.title('Evolución del costo en el gradiente descendente')
plt.axhline(y=0, color='r', alpha=0.7, linestyle='dashed')
plt.legend()
plt.show()

# Generar datos para el gráfico de dispersión
x_svd_norm = np.linalg.norm(x_svd)
x_f_norm = np.linalg.norm(x_f)
x_f2_norm = np.linalg.norm(x_f2)
x_f1_norm = np.linalg.norm(x_f1)

x_labels = ['SVD', 'F', 'F2', 'F1']
x_norm_values = [x_svd_norm, x_f_norm, x_f2_norm, x_f1_norm]

# Crear el gráfico de dispersión
plt.scatter(x_labels, x_norm_values)
plt.xlabel('Método')
plt.ylabel('Norma de x')
plt.title('Comparación de la norma de x')
plt.show()

# Grafico de las normas 2 de la evolucion de los gradientes
plt.plot(iterations, gradients_F, label='F')
plt.plot(iterations, gradients_F2, label='F2')
plt.plot(iterations, gradients_F1, label='F1')
plt.xlabel('Iteraciones')
plt.ylabel('Norma 2 de la solución')
plt.ylim(bottom=-1, top=10)
plt.title('Evolución de la norma 2 del gradiente de F F1 y F2')
plt.show()



"""EJERCICIO 2"""


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

# Ejemplo de uso
m = 100
n = 20
condition_number = 8 # Número de condición deseado
A = generate_random_matrix_with_condition(m, n, condition_number)
print(f"Número de condición de A: {np.linalg.cond(A)}")

def gradient_descent_iterations(A, b, cond_number, tol):
    n = A.shape[1]  # Número de columnas de A
    max_eigval = cond_number**2  # Autovalor máximo (asumiendo número de condición dado)
    s = 1 / max_eigval  # Tamaño del paso

    x = np.random.rand(n, 1)  # Condición inicial aleatoria
    error = np.real(cost_function(A, x, b))[0][0]
    iterations = 0

    while error > tol:
        gradient = cost_function_gradient(A, b, x)
        x = x - s * gradient
        error = np.real(cost_function(A, x, b))[0][0]
        iterations += 1

        print(f"Iteración {iterations}: {error}")

    return iterations

tol = 1e-2

condition_numbers = [20]  # Números de condición de A

# Calcular número de iteraciones y comparar con predicción teórica
for cond_number in condition_numbers:
    A = generate_random_matrix_with_condition(m, n, cond_number)

    b = np.random.rand(m, 1)  # Vector b aleatorio

    predicted_iterations = np.log(1 / tol) / np.log(cond_number)
    actual_iterations = gradient_descent_iterations(A, b, cond_number, tol)

    print(f"Número de condición de A: {np.linalg.cond(A)}")
    print(f"Número de iteraciones (predicción teórica): {predicted_iterations}")
    print(f"Número de iteraciones (obtenido): {actual_iterations}")
    print("---------")



