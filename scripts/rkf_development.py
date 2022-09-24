"""
Script for working out how to make the RKF45 method work.
"""
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

tol = 1E-6


def main(f):
    h = 1
    y0 = 1
    t0 = 0
    t = t0
    y = y0

    p = 4
    beta = 0.9

    ts = [t0]
    ys = [y0]

    while t <= 10:
        step_accepted = False
        while not step_accepted:
            ykplus1, zkplus1 = step(f, h, t, y)
            epsilon = abs(zkplus1 - ykplus1) / ((2 ** p) - 1)
            if epsilon < tol:
                step_accepted = True
                h_new = beta * h * ((tol / epsilon) ** (1 / p))
            else:
                h_new = beta * h * ((tol / epsilon) ** (1 / (p + 1)))
                h = h_new

        y = ykplus1
        t += h
        ts.append(t)
        ys.append(y)
        h = h_new

    pprint(ts)
    pprint(ys)
    plt.scatter(ts, ys, label="mine")
    #plt.plot(ts, np.tan(ts), label="tan")
    plt.legend()
    plt.show()

# https://www.uni-muenster.de/imperia/md/content/physik_tp/lectures/ss2017/numerische_Methoden_fuer_komplexe_Systeme_II/rkm-1.pdf


def step(f, h, tk, yk):
    k1 = h * f(tk,             yk)
    k2 = h * f(tk +   (1/4)*h, yk +       (1/4)*k1)
    k3 = h * f(tk +   (3/8)*h, yk +      (3/32)*k1 +      (9/32)*k2)
    k4 = h * f(tk + (12/13)*h, yk + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
    k5 = h * f(tk +         h, yk +   (439/216)*k1 -           8*k2 +  (3680/513)*k3 -  (845/4104)*k4)
    k6 = h * f(tk +   (1/2)*h, yk -      (8/27)*k1 +           2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)

    ykplus1 = yk + (25/216)*k1 + (1408/2565)*k3 + (2197/4101)*k4 - (1/5)*k5
    zkplus1 = yk + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6

    return ykplus1, zkplus1


def func(x, y1):
    #return (-2 * y1) + np.exp(-2 * ((x - 6) ** 2))
    return np.sin(x)


if __name__ == "__main__":
    main(func)
