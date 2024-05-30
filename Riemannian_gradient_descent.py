# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:41:03 2024

@author: WONCHAN
"""

import numpy as np
import matplotlib.pyplot as plt

# Riemannian Gradient Descent Algorithm Implementation
def riemannian_gradient_descent(x0, x_star, alpha, epsilon, max_iter, tolerance):
    def compute_gradient(x):
        return 2 * (x - x_star)
    
    def retraction(x, v):
        return x + v

    def distance(x, y):
        return np.linalg.norm(x - y)
    
    k = 0
    x_k = x0
    distances = [distance(x_k, x_star)]
    initial_alpha = alpha
    
    while k < max_iter:
        g_mu = compute_gradient(x_k)
        alpha = initial_alpha / (1 + 0.001 * k)  # Gradually decreasing learning rate
        x_k1 = retraction(x_k, -alpha * g_mu)
        
        distances.append(distance(x_k1, x_star))
        
        if np.linalg.norm(g_mu) < epsilon:
            print(f"Stopping criterion met at iteration {k}: gradient norm < epsilon")
            break
        
        if distances[-1] < tolerance:
            print(f"Stopping criterion met at iteration {k}: distance to x* is below tolerance")
            break
        
        x_k = x_k1
        k += 1
    
    return x_k, distances

# Parameters
initial_point = np.array([10.0, 10.0])
x_star = np.array([0.0, 0.0])
alpha = 0.001  # Initial learning rate
epsilon = 1e-2
max_iter = 1000000
tolerance = 0

# Run Riemannian Gradient Descent
converged_point, distances = riemannian_gradient_descent(initial_point, x_star, alpha, epsilon, max_iter, tolerance)

# Print results
print("Converged point:", converged_point)
print("Final distance to x*:", distances[-1])

# Plot distance to x* over iterations in log scale
plt.figure()
plt.semilogy(distances, marker='o')
plt.plot(distances, 'r-')
plt.xlabel('Iteration')
plt.ylabel('Distance to x* (log scale)')
plt.title('Distance to x* over iterations (Riemannian Gradient Descent)')
plt.grid(True)
plt.show()
