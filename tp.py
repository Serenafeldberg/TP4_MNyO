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


# """GRAFICO DE RESULTADOS"""
# plt.plot(x_svd, label='SVD')
# plt.plot(x_f, label='F')
# plt.plot(x_f2, label='F2')
# plt.plot(x_f1, label='F1')
# plt.legend()
# plt.xlabel('vector x')
# plt.ylabel('valor de x')
# plt.title('Comparación de soluciones')
# plt.show()

# """GRAFICO DE RESULTADOS"""
# # Generar datos para el gráfico de dispersión
# norm_svd = np.linalg.norm(x_svd, 2)
# norm_f = np.linalg.norm(x_f, 2)
# norm_f2 = np.linalg.norm(x_f2, 2)
# norm_f1 = np.linalg.norm(x_f1, 2)

# x_labels = ['SVD', 'F', 'F2', 'F1']
# x_norm_values = [norm_svd, norm_f, norm_f2, norm_f1]

# plt.scatter(x_labels, x_norm_values)
# plt.xlabel('Método')
# plt.ylabel('Norma 2 de la solución')
# plt.title('Comparación de las normas 2 de las soluciones')
# plt.show()


"""GRAFICO DELTAS"""
delta2 = delta2(A)
delta2_low = (10**(-3))* (np.linalg.eigvals(2*A.T.dot(A)).max())
delta2_high = (10**(-1))* (np.linalg.eigvals(2*A.T.dot(A)).max())

delta_1 = delta1(A)
delta1_low = (10**(-4))* (np.linalg.eigvals(2*A.T.dot(A)).max())
delta1_high = (10**(-2))* (np.linalg.eigvals(2*A.T.dot(A)).max())

x_f2_low, results2_low, gradients_F2_low = regularized_gradient_descent(A, b, s, delta2_low, 1000, x0)  # Solución minimizando F2
x_f2_high, results2_high, gradients_F2_high = regularized_gradient_descent(A, b, s, delta2_high, 1000, x0)  # Solución minimizando F2

x_f1_low, results3_low, gradients_F1_low = regu2_gradient_descent(A, b, s, delta1_low, 1000, x0)  # Solución minimizando F1
x_f1_high, results3_high, gradients_F1_high = regu2_gradient_descent(A, b, s, delta1_high, 1000, x0)  # Solución minimizando F1

fig, axes = plt.subplots(1, 2)

# axes[0].plot(results2_low.flatten(), label='F2 delta bajo')
# axes[0].plot(results2.flatten(), label='F2 delta medio')
# axes[0].plot(results2_high.flatten(), label='F2 delta alto')
# axes[0].set_title('F2')
# axes[0].set_xlabel('Iteraciones')
# axes[0].set_ylim(-1, 40)

# axes[1].plot(results3_low.flatten(), label='F1 delta bajo')
# axes[1].plot(results3.flatten(), label='F1 delta medio')
# axes[1].plot(results3_high.flatten(), label='F1 delta alto')
# axes[1].set_title('F1')
# axes[1].set_xlabel('Iteraciones')
# axes[1].set_ylim(-1, 40)

# axes[0].legend()
# axes[1].legend()

axes[0].plot(x_f2_low, label='F2 delta bajo')
axes[0].plot(x_f2, label='F2 delta medio')
axes[0].plot(x_f2_high, label='F2 delta alto')
axes[0].plot(x_svd, label='SVD')
axes[0].set_title('F2')
axes[0].set_xlabel('vector x')
axes[0].set_ylabel('valor de x')
axes[0].set_ylim(-0.5, 0.8)

axes[1].plot(x_f1_low, label='F1 delta bajo')
axes[1].plot(x_f1, label='F1 delta medio')
axes[1].plot(x_f1_high, label='F1 delta alto')
axes[1].plot(x_svd, label='SVD')
axes[1].set_title('F1')
axes[1].set_xlabel('vector x')
axes[1].set_ylabel('valor de x')
axes[1].set_ylim(-0.5, 0.8)

axes[0].legend()
axes[1].legend()

plt.show()

"""DELTA MEDIO Y SVD"""

fig, ax = plt.subplots(1, 2)

ax[0].plot (x_svd, label='SVD')
ax[0].plot (x_f2, label='F2')
ax[0].set_title('F2')
ax[0].set_xlabel('vector x')
ax[0].set_ylabel('valor de x')
ax[0].set_ylim(-0.5, 0.8)

ax[1].plot (x_svd, label='SVD')
ax[1].plot (x_f1, label='F1')
ax[1].set_title('F1')
ax[1].set_xlabel('vector x')
ax[1].set_ylabel('valor de x')
ax[1].set_ylim(-0.5, 0.8)

ax[0].legend()
ax[1].legend()

plt.show()

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




