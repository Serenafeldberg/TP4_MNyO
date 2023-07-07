import numpy as np
import matplotlib.pyplot as plt


def cost (A, x, b):
    return (A @ x - b).T @ (A @ x - b)   

def grad(A, x, b):
    return 2 * A.T @ (A @ x - b)

def grad_descent(A, b, x0, step, max_iter, tolerance):
    errors = []
    i = 0
    x = x0
    gradient_x = grad(A, x, b)
    errors.append(cost(A, x, b)[0][0])
    while (cost(A,x,b)[0][0] > tolerance) and i <= max_iter:
        x = x - step * gradient_x
        gradient_x = grad(A, x, b)
        errors.append(cost(A, x, b)[0][0])
        i += 1
    return x, i, errors

def create_conditioned_matrix(A, condition_n):
    U, S, Vt = np.linalg.svd(A)
    S[0] = condition_n 
    for i in range (1, len(S)): S[i] = 1
    S = np.diag(S)
    matriz_ajustada = U @ S @ Vt
    return matriz_ajustada

def iterations_with_diferent_conditions():
    m = 100
    n = 100
    tol = 10e-2
    max_iter = 100000
    iteration_counts = []
    cond_ns = range(1, 100, 1)
    A = np.random.rand(m, n)
    for cond_n in cond_ns:
        A = create_conditioned_matrix(A, cond_n)
        b = np.random.rand(m, 1)
        x0 = np.random.rand(n, 1)
        lambda_max = np.real(np.max(np.linalg.eigvals(2 * A.T @ A)))
        s = 1 / lambda_max
        _, iterations, _ = grad_descent(A, b, x0, s, max_iter, tol)
        iteration_counts.append(iterations)

    # Gráfico de iteraciones vs. números de condición
    plt.semilogy(cond_ns, iteration_counts, 'bo-')
    plt.xlabel('Número de Condición')
    plt.ylabel('Iteraciones')
    plt.title('Iteraciones para diferentes números de condición')
    plt.show()

def analyze_convergence():
    m = 100
    n = 100
    tol = 10e-2
    max_iter = 100000
    cond_n = 1
    A = np.random.rand(m, n)
    A = create_conditioned_matrix(A, cond_n)
    b = np.random.rand(m, 1)
    x0 = np.random.rand(n,1)
    lambda_max = np.real(np.max(np.linalg.eigvals(2*A.T @ A)))
    s = 1/lambda_max
    _, _, errors = grad_descent(A, b, x0, s, max_iter, tol)

    # Gráfico del error cuadrático medio a lo largo de las iteraciones
    iterations = range(len(errors))
    plt.plot(iterations, errors, 'b-')
    plt.xlabel('Iteraciones')
    plt.ylabel('Error')
    plt.title('Convergencia del Gradiente Descendente')
    plt.show()

def main():
    m = 100
    n = 100
    tol = 10e-2
    max_iter = 100000
    cond_n = 1
    A = np.random.rand(m, n)
    A = create_conditioned_matrix(A, cond_n)
    b = np.random.rand(m, 1)
    x0 = np.random.rand(n,1)
    lambda_max = np.real(np.max(np.linalg.eigvals(2*A.T @ A)))
    s = 1/lambda_max
    x , i, results = grad_descent(A, b, x0, s, max_iter, tol)
    print("Ax:")
    print(A@x)
    print("b:")
    print(b)

# main()
iterations_with_diferent_conditions()
analyze_convergence()