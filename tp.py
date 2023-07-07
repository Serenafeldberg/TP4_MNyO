import numpy as np
import matplotlib.pyplot as plt

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

# print("Solución mediante SVD:")
# print(x_svd[:3])
# print("Costo (F):", cost_svd)
# print()
# print("Solución minimizando F:")
# print(x_f[:3])
# print("Costo (F):", cost_f)
# print()
# print("Solución minimizando F2:")
# print(x_f2[:3])
# print("Costo (F2):", cost_f2)
# print()
# print("Solución minimizando F1:")
# print(x_f1[:3])
# print("Costo (F1):", cost_f1)

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
error_f = cost_function(A, x_f, b)
error_f2 = cost_function(A, x_f2, b)
error_f1 = cost_function(A, x_f1, b)
error_svd = cost_function(A, x_svd, b)

x_labels = ['SVD', 'F', 'F2', 'F1']
x_norm_values = [error_svd, error_f, error_f2, error_f1]

# Crear el gráfico de dispersión
plt.scatter(x_labels, x_norm_values)
plt.xlabel('Método')
plt.ylabel('Costo')
plt.title('Comparación de los costos de los métodos')
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

s_low = 0.00007
s_medium = learning_rate(A)
print(s_medium)
s_high = 0.01
x_low,cost_history_F_low, grF = gradient_descent(A, b, s_low, 1000, x0)
x_medium, cost_history_F_medium, grF2 = gradient_descent(A, b, s_medium, 1000, x0)
x_high, cost_history_F_high, grF1 = gradient_descent(A, b, s_high, 1000, x0)

x_low2, cost_history_F2_low, grF_ = regularized_gradient_descent(A, b, s_low, delta2_value, 1000, x0)
x_medium2, cost_history_F2_medium, grF2_ = regularized_gradient_descent(A, b, s_medium, delta2_value, 1000, x0)
x_high2, cost_history_F2_high, grF1_ = regularized_gradient_descent(A, b, s_high, delta2_value, 1000, x0)

x_low3, cost_history_F1_low, grF__ = regu2_gradient_descent(A, b, s_low, delta1_value, 1000, x0)
x_medium3, cost_history_F1_medium, grF2__ = regu2_gradient_descent(A, b, s_medium, delta1_value, 1000, x0)
x_high3, cost_history_F1_high, grF1__ = regu2_gradient_descent(A, b, s_high, delta1_value, 1000, x0)

# Crear figura y subfiguras
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Subfigura con learning rate bajo
axes[0].plot(cost_history_F_low.flatten(), label='s alto')
axes[0].plot(cost_history_F_medium.flatten(), label='s medio')
axes[0].plot(cost_history_F_high.flatten(), label='s alto')
axes[0].set_xlabel('Iteraciones')
axes[0].set_ylabel('Costo')
axes[0].set_title('F')
axes[0].legend()
axes[0].set_ylim(-1, 50)  # Establecer límite del eje y

# Subfigura con learning rate medio
axes[1].plot(cost_history_F2_low.flatten(), label='s bajo')
axes[1].plot(cost_history_F2_medium.flatten(), label='s medio')
axes[1].plot(cost_history_F2_high.flatten(), label='s alto')
axes[1].set_xlabel('Iteraciones')
axes[1].set_ylabel('Costo')
axes[1].set_title('F2')
axes[1].legend()
axes[1].set_ylim(-1, 50)  # Establecer límite del eje y

# Subfigura con learning rate alto
axes[2].plot(cost_history_F1_low.flatten(), label='s bajo')
axes[2].plot(cost_history_F1_medium.flatten(), label='s medio')
axes[2].plot(cost_history_F1_high.flatten(), label='s alto')
axes[2].set_xlabel('Iteraciones')
axes[2].set_ylabel('Costo')
axes[2].set_title('F1')
axes[2].legend()
axes[2].set_ylim(-1, 50)  # Establecer límite del eje y

# Ajustar espacio entre subfiguras
plt.subplots_adjust(wspace=0.3)

# Mostrar gráfico
plt.show()



"""EJERCICIO 2"""

# def generate_random_matrix_with_condition(m, n, condition_number):
#     # Generar una matriz aleatoria A de tamaño m x n
#     A = np.random.rand(m, n)

#     # Calcular la descomposición SVD
#     U, S, Vt = np.linalg.svd(A, full_matrices=False)

#     # Ajustar los valores singulares según el número de condición deseado
#     desired_condition = condition_number
#     current_condition = np.max(S) / np.min(S)
#     S_modified = S * (desired_condition / current_condition) ** 0.5

#     # Reconstruir la matriz A con los valores singulares modificados
#     A_modified = U @ np.diag(S_modified) @ Vt

#     return A_modified

# # Ejemplo de uso
# m = 100
# n = 100
# condition_number = 10  # Número de condición deseado
# A = generate_random_matrix_with_condition(m, n, condition_number)
# s = learning_rate(A)

# def gradient_descent_iterations(A, b, cond_number, tol, s):
#     n = A.shape[1]  # Número de columnas de A

#     x = np.random.rand(n, 1)  # Condición inicial aleatoria
#     error = float('inf')
#     iterations = 0

#     while error > tol:
#         gradient = cost_function_gradient(A, x, b)
#         x -= s * gradient
#         error = cost_function(A, x, b)
#         iterations += 1

#     return iterations

# tol = 1e-2

# condition_numbers = [1]  # Números de condición de A

# # Calcular número de iteraciones y comparar con predicción teórica
# for cond_number in condition_numbers:
#     A = generate_random_matrix_with_condition(m, n, cond_number)

#     b = np.random.rand(m, 1)  # Vector b aleatorio

#     predicted_iterations = np.log(1 / tol) / np.log(cond_number)
#     actual_iterations = gradient_descent_iterations(A, b, cond_number, tol)

#     print(f"Número de condición de A: {cond_number}")
#     print(f"Número de iteraciones (predicción teórica): {predicted_iterations}")
#     print(f"Número de iteraciones (obtenido): {actual_iterations}")
#     print("---------")




