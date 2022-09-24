"""
Implement a 1D RKF just to see if it works.
Update: It does.
"""
import numpy as np


def rhs(t, X):
    return np.array([X[1], t], dtype=np.float64)


def main():
    init = np.array([0, 0], dtype=np.float64)
    X = np.array([0, 0], dtype=np.float64)

    t = 0
    tMax = 5
    dt = 0.1

    nSteps = round((tMax - t) / dt)
    for stepNumber in range(1, nSteps + 1):
        k1 = rhs(t, X)
        k2 = rhs(t+(dt/2), X+(k1 * (dt / 2)))
        k3 = rhs(t+(dt/2), X+(k2 * (dt / 2)))
        k4 = rhs(t+dt, X+dt*k3)

        X += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt

    print(f"Final time: {t}")
    print(f"Final position: {X[0]}")
    print(f"Final velocity: {X[1]}")


if __name__ == "__main__":
    main()
