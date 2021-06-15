import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from model import objective_1, constraints_1
from steepest_descent import steepest_descent, random_descent

#s_old = [0, 0.2, 0.1, 0, 0, 0, 0.2, 0.1, 0, 0, 0, 0, 0.1, 0.2, 0.2, 0, 0, 0, 0, 0, 0.3, 0, 0, 0.4, 0, -0.4, 0, 0.2, 0.2, 0.1, 0, 0]
s_default = [0, 0.1, 0.1, 0.1, 0, 0.05, 0.1, 0.1, 0.05, 0, 0, 0, 0, 0.5, 0, 0.1, 0, 0, 0, 0, 0.3, 0, 0.05, 0.25, 0, -0.25, 0.1, 0.2, 0 ]
g_default = [1, 1,   1,   1,   1, 1,    1,   1,   1,    1, 1, 1, 1, 0  , 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 1,     1,  1,   1,   1,   1  ]


# basic optimization of a track segment with given length
def optimization_0(accuracy=1, length=29, a_max=16, gas=False, method="Nelder-Mead"):
    t0 = time.time()
    N = length*accuracy

    plt.plot(0, 0, c="r", linestyle="-")  # blank plots used for defining legend only
    plt.plot(0, 0, c="b")

    # extend initial guess vector by duplicating entries for accuracy > 1
    s_vec, g_vec = [], []
    for i in range(len(s_default)):
        for j in range(accuracy):
            s_vec.append(s_default[i])
            if gas:
                g_vec.append(g_default[i])
    x = s_vec[0:N] + g_vec[0:N]

    f0 = objective_1(x, [[], [], 1 / accuracy, a_max, gas, False], plot=True, style="v")

    # constrained optimization
    if method in ["COBYLA", "SLSQP", "trust-constr"]:
        options = {"rhobeg": 0.01}
        #options = {}
        if gas:
            bounds = [(-0.3, 0.3)]*int(len(x)/2) + [(0, 1)]*int(len(x)/2)
        else:
            bounds = [(-0.5, 0.5)]*len(x)
        args = [[], [], 1 / accuracy, a_max, gas, True]
        con = [{"type": "ineq", "fun": constraints_1, "args": [args]}]
        out = minimize(objective_1, args=args, x0=x, options=options, method=method,
                       bounds=bounds, constraints=con)

    # optimization using penalty functions
    elif method in ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC"]:
        args = [[], [], 1 / accuracy, a_max, gas, False]
        options = {"maxiter": 10000, "adaptive": True, "disp": True}
        out = minimize(objective_1, args=args,  x0=x, options=options, method=method)

    # optimization using custom steepest descent algorithm
    elif method == "SGD":
        args = [[], [], 1 / accuracy, a_max, gas, False]
        out = steepest_descent(objective_1, x, args=args, messages=True, max_iter=100, f_tol=1e-8, g_tol=1e-8, fd_step_size=1e-5)

    elif method == "RGD":
        args = [[], [], 1 / accuracy, a_max, gas, False]
        out = random_descent(objective_1, x, args=args, messages=True, max_iter=1000)

    print("Optimization time:", str(time.time() - t0), "seconds")
    # plot solution
    print(out)
    if not method in ["SGD", "RGD"]:
        out = out.x
    f1 = objective_1(out, [[], [], 1 / accuracy, a_max, gas, False], plot=True, style="v")
    print("n_laps/n_laps_0:", f1/f0)
    plt.legend(["Initial guess", "Optimized result"])
    plt.show()

# Run optimizer in a loop for progressively longer vectors to solve early track segments more easily
def optimization_1(n, accuracy=1, length=30, a_max=16, gas=False, method="Nelder-Mead"):
    t0 = time.time()
    N = length*accuracy

    # extend initial guess vector by duplicating entries for accuracy > 1
    s_vec, g_vec = [], []
    for i in range(len(s_default)):
        for j in range(accuracy):
            s_vec.append(s_default[i])
            if gas:
                g_vec.append(g_default[i])

    # evaluate initial objective
    f0 = objective_1(s_vec[0:N] + g_vec[0:N], [[], [], 1 / accuracy, a_max, gas, False])
    print(f0)

    out_vec = []
    # run optimizations in a loop
    for i in range(N-1):

        # construct design vector and start vector which is not optimized
        if gas:  # optimization of both gas and steering vectors
            out_s, out_g = out_vec[:int(len(out_vec)/2)], out_vec[int(len(out_vec)/2):]
            if i == 0:  # in first round use first two entries of guess vector
                x = s_vec[:2] + g_vec[:2]
                s_i, g_i = [], []
            elif 0 < i < n-1:  # vector is growing to intended length
                x = out_s + [s_vec[len(out_s)]] + out_g + [g_vec[len(out_g)]]
                s_i, g_i = [], []
            else:  # vector of desired length moving down track
                s_i, g_i, x = [], [], []
                for j in range(i+2):  # loop over steering vector
                    if j <= i - n + 1:
                        s_i.append(out_s[j])  # construct initial vector
                    elif j == i + 1:
                        x.append(s_vec[j])  # add new guess to the end of design vector
                    else:
                        x.append(out_s[j])  # construct design vector
                for j in range(i+2):  # loop over gas vector
                    if j <= i - n + 1:
                        g_i.append(out_g[j])  # construct initial vector
                    elif j == i + 1:
                        x.append(g_vec[j])  # add new guess to the end of design vector
                    else:
                        x.append(out_g[j])  # construct design vector

        else:  # optimization of steering only
            g_i = []
            if i == 0:  # in first round use first two entries of guess vector
                x = s_vec[:2]
                s_i = []
            elif 0 < i < n-1:  # vector is growing to intended length
                x = out_vec + [s_vec[len(out_vec)]]
                s_i = []
            else:  # vector of desired length moving down track
                s_i, x = [], []
                for j in range(i+2):
                    if j <= i - n + 1:
                        s_i.append(out_vec[j])
                    elif j == i + 1:
                        x.append(s_vec[j])
                    else:
                        x.append(out_vec[j])

        print("Optimization ", str(i + 1) + "/" + str(N - 1))

        # constrained optimization
        if method in ["COBYLA", "SLSQP", "trust-constr"]:
            if method == "COBYLA":
                options = {"rhobeg": 0.01}
            else:
                options = {}
            if gas:
                bounds = [(-0.3, 0.3)]*int(len(x)/2) + [(0, 1)]*int(len(x)/2)
            else:
                bounds = [(-0.3, 0.3)]*len(x)
            args = [s_i, g_i, 1 / accuracy, a_max, gas, True]
            con = [{"type": "ineq", "fun": constraints_1, "args": [args]}]
            out = minimize(objective_1, args=args, x0=x, options=options, method=method,
                           bounds=bounds, constraints=con)
            x_out = out.x

        # optimization using penalty functions
        elif method in ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC"]:
            options = {"maxiter": 10000, "adaptive": True}
            args = [s_i, g_i, 1 / accuracy, a_max, gas, False]
            out = minimize(objective_1, args=args,  x0=x, options=options, method=method)
            x_out = out.x

        # optimization using custom steepest descent algorithm
        elif method == "SGD":
            args = [s_i, g_i, 1 / accuracy, a_max, gas, False]
            out = steepest_descent(objective_1, x, args=args, messages=True, max_iter=100)
            x_out = out

        elif method == "RGD":
            args = [s_i, g_i, 1 / accuracy, a_max, gas, False]
            out = random_descent(objective_1, x, args=args, messages=False, max_iter=1000)
            x_out = out

        if gas:
            out_vec = s_i + list(x_out)[:int(len(x_out)/2)] + g_i + list(x_out)[int(len(x_out)/2):]
        else:
            out_vec = s_i + list(x_out)

    print("Optimization time:", str(time.time() - t0), "seconds")
    # plot solution
    # print(g_i + list(x_out)[int(len(x_out)/2):])
    print(out_vec)

    f1 = objective_1(out_vec, [[], [], 1 / accuracy, a_max, gas, False], plot=True, style="v")
    print("n_laps/n_laps_0:", f1 / f0)
    print(f1)
    plt.show()


#     0              1         2      3        4          5       6        7           8           9      10
m = ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "trust-constr", "SGD", "RGD"]

optimization_1(10, accuracy=2, length=29, a_max=13, gas=True, method=m[10])

#optimization_0(accuracy=1, length=29, a_max=7, gas=True, method=m[10])

