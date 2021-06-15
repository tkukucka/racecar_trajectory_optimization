import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import golden
from numdifftools import Hessian

def test_function(x, args):
    x1 = x[0]
    x2 = x[1]
    return x1**2 + 4*x1 + 0.5*x2**2 - 2*x2 + 3

# call objective along a line in search direction - function wrapper used by golden search
def search_function(alpha, func, direction, x0, args, return_x=False):
    x = []
    for i in range(len(direction)):
        x.append(x0[i] + alpha*direction[i])
    if return_x:
        return x
    else:
        return func(x, args)

def steepest_descent(func, x0, args=None, fd_step_size=1e-8, max_iter=100, g_tol=1e-7, f_tol=1e-9, messages=False, return_history=False):
    n = len(x0)
    x = x0
    found = False
    x_history, f_history = [x], []
    n_f_eval = 0

    # repeat optimization steps in a loop
    for loop in range(max_iter):
        f0 = func(x, args)
        if loop == 0:
            f_history.append(f0)
        n_f_eval += 1

        direction = []
        s = 0

        # compute gradients in a loop
        for j in range(n):
            x[j] = x[j] + fd_step_size  # make perturbation in x
            f = func(x, args)
            n_f_eval += 1
            d = (f0 - f) / fd_step_size  # compute derivative *-1
            direction.append((f0 - f) / fd_step_size)
            s += d**2
            x[j] = x[j] - fd_step_size  # fix perturbation in x

        dir_length = s**0.5
        zero_grad = True
        if dir_length > g_tol:  # check for convergence
            zero_grad = False
            # normalize search direction vector
            for j in range(n):
                direction[j] = direction[j]/dir_length

            # apply golden search in steepest descent direction
            args2 = func, direction, x, args
            alpha, val, nfe = golden(search_function, args=args2, full_output=True)
            n_f_eval += nfe
            x = search_function(alpha, func, direction, x, args, return_x=True)
            f1 = func(x, args)
            n_f_eval += 1

            if messages:
                print("Iteration", loop, f1)
            x_history.append(x)
            f_history.append(f1)

        no_change = False
        # Check if function value changed in last two iterations
        if loop > 1:
            if abs(f_history[-1] - f_history[-3]) < f_tol:
                no_change = True

        if zero_grad or no_change:
            found = True
            pos_def = True

            # Check for positive definiteness
            hess = Hessian(func, step=fd_step_size)(x, args)
            n_f_eval += n**2*3  # function evaluations for evaluating Hessian
            eig = np.linalg.eig(hess)[0]
            for e in eig:
                if e < 0:
                    pos_def = False

            if messages:
                if pos_def:
                    print("Minimum found")
                    print("x =", x)
                    print("f =", f0)
                else:
                    print("OPTIMALITY CONDITIONS NOT SATISFIED")
            break

    if not found:
        print("Number of iterations exceeded maximum value")
    print("Number of function evaluations:", n_f_eval)
    if return_history:
        return x_history
    else:
        return x

def random_descent(func, x0, args=None, fd_step_size=1e-8, max_iter=100, messages=False, return_history=False):
    n = len(x0)
    x = x0
    x_history, f_history = [x], []
    n_f_eval = 0

    # repeat optimization steps in a loop
    for loop in range(max_iter):
        f0 = func(x, args)
        if loop == 0:
            f_history.append(f0)
        n_f_eval += 1

        # construct random search direction
        s = 0
        #direction = np.random.uniform(-1, 1, n)
        direction = np.zeros(n)
        for i in range(3):
            direction[random.randint(0, n-1)] = random.uniform(-1, 1)

        for d in direction:
            s += d**2
        dir_length = s**0.5
        for j in range(n):
            direction[j] = direction[j] / dir_length

        # check if search direction gradient is negative
        x1 = []
        for i in range(n):
            x1.append(x[i]+direction[i]*fd_step_size)
        f1 = func(x1, args)
        #print(direction)

        if f1 < f0:

            # apply golden search in steepest descent direction
            args2 = func, direction, x, args
            alpha, val, nfe = golden(search_function, args=args2, full_output=True)
            n_f_eval += nfe
            x = search_function(alpha, func, direction, x, args, return_x=True)
            f1 = func(x, args)
            n_f_eval += 1

            if messages:
                print("Iteration", loop, f1)
            x_history.append(x)
            f_history.append(f1)

    if return_history:
        return x_history
    else:
        return x

def make_plot(func):
    X1 = np.linspace(-4, 2, 50)
    X2 = np.linspace(-2.5, 3, 50)

    X1, X2 = np.meshgrid(X1, X2)
    F = func([X1, X2], None)

    plt.contour(X1, X2, F, 15, colors='black')
    plt.contourf(X1, X2, F, 100, cmap='bwr')

    x_history = steepest_descent(func, [1, -2], messages=True, return_history=True)
    x1, x2 = [], []
    for x in x_history:
        x1.append(x[0])
        x2.append(x[1])
    plt.plot(x1, x2, c="gray")
    plt.colorbar(label="function value")
    plt.axis("equal")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.show()

if __name__ == "__main__":
    make_plot(test_function)
    #steepest_descent(test_function, [1, 1], messages=True)

