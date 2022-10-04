import os

import numpy as np
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

NUM_SIMULATIONS = 100

LOCKBOX_SIZE = 20

# Limit where to stop simulation
LIMIT = 5

# Time step size
STEP_SIZE = 0.001

# Total time
TOTAL_TIME = 10

# Time vector
STEPS = int(TOTAL_TIME / STEP_SIZE)

# Body mass
M1 = 1
M2 = 1
M3 = 1
M = np.array([M1, M2, M3])

# Newton's gravitational constant
G = 1


def initialize_positions():
    x10 = -1
    y10 = 0

    x20 = 1
    y20 = 0

    x30 = 0
    y30 = 0

    return np.array([[x10, y10], [x20, y20], [x30, y30]])


def create_initial_velocities():
    v1 = random.uniform(0, 1)
    v2 = random.uniform(0, 1)

    return v1, v2


def initialize_velocities(v1, v2):
    vx10 = v1
    vy10 = v2

    vx20 = v1
    vy20 = v2

    vx30 = -2 * v1
    vy30 = -2 * v2

    return np.array([[vx10, vy10], [vx20, vy20], [vx30, vy30]])


def initialize_arrays():
    # Initial conditions - positions
    positions = initialize_positions()

    # Initial conditions - velocities
    v1, v2 = create_initial_velocities()
    velocities = initialize_velocities(v1, v2)

    # Prepare vectors to store the solution for n bodies
    x = np.zeros((STEPS, 3))
    y = np.zeros((STEPS, 3))
    vx = np.zeros((STEPS, 3))
    vy = np.zeros((STEPS, 3))

    # Assign the initial condition to the first element of the solution vectors
    for i in range(3):
        x[0, i] = positions[i, 0]
        y[0, i] = positions[i, 1]

        vx[0, i] = velocities[i, 0]
        vy[0, i] = velocities[i, 1]

    return x, y, vx, vy


def compute_acceleration(x, y):
    ax_tot = np.zeros(3)
    ay_tot = np.zeros(3)

    for j in range(3):
        dx = x[j] - x
        dy = y[j] - y

        test_x = np.zeros(3)
        test_y = np.zeros(3)

        for i in range(3):
            if dx[i] == dy[i] == 0:
                test_x[i] = 0
                test_y[i] = 0
            else:
                test_x[i] = (-dx[i] * M[i] * G) / (np.sqrt(dx[i] ** 2 + dy[i] ** 2)) ** 3
                test_y[i] = (-dy[i] * M[i] * G) / (np.sqrt(dx[i] ** 2 + dy[i] ** 2)) ** 3

        ax_tot[j] = sum(test_x)
        ay_tot[j] = sum(test_y)

    return ax_tot, ay_tot


def verlet_ode_solver(x, y, vx, vy, i):
    for step in range(STEPS - 1):
        ax_tot, ay_tot = compute_acceleration(x[step], y[step])
        x[step + 1] = x[step] + STEP_SIZE * vx[step] + (1 / 2) * (ax_tot * (STEP_SIZE ** 2))
        y[step + 1] = y[step] + STEP_SIZE * vy[step] + (1 / 2) * (ay_tot * (STEP_SIZE ** 2))

        ax_tot_next, ay_tot_next = compute_acceleration(x[step + 1], y[step + 1])
        vx[step + 1] = vx[step] + (1 / 2) * (ax_tot + ax_tot_next) * STEP_SIZE
        vy[step + 1] = vy[step] + (1 / 2) * (ay_tot + ay_tot_next) * STEP_SIZE

        if max(abs(x[step + 1])) > LIMIT or max(abs(y[step + 1])) > LIMIT:
            x = x[0:step]
            y = y[0:step]
            vx = vx[0:step]
            vy = vy[0:step]
            break

    df = pd.DataFrame(np.hstack((x, y, vx, vy)))
    df.to_csv(os.getcwd() + "/data/raw/simulation_" + str(i) + ".csv", header=False, index=False)

    return x, y, vx, vy


def print_planets(x, y, i):
    matplotlib.use('TkAgg')

    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(x[:, 0], y[:, 0], color='#B00B69', label="Body 1", linestyle="dotted")
    ax.plot(x[:, 1], y[:, 1], color='#420A55', label="Body 2", linestyle="dashed")
    ax.plot(x[:, 2], y[:, 2], color='#042069', label="Body 3")

    ax.set(xlabel='x', ylabel='y', title='Simulation' + str(i))
    plt.xlim(-(LIMIT+1), LIMIT+1)
    plt.ylim(-(LIMIT+1), LIMIT+1)

    ax.legend()
    ax.grid()

    fig.savefig(os.getcwd() + "/data/plots/simulation_" + str(i) + ".png")
    # plt.show()
    plt.close()


def normalize_simulation(x, y, vx, vy, i):
    # normalize the positions by dividing with the coordinate threshold limit
    x = x / LIMIT
    y = y / LIMIT
    # normalize the velocities by dividing with the square of the coordinate threshold limit
    vx = vx / (LIMIT ** 2)
    vy = vy / (LIMIT ** 2)

    df = pd.DataFrame(np.hstack((x, y, vx, vy)))
    df.to_csv(os.getcwd() + "/data/processed/simulation_" + str(i) + ".csv", header=False, index=False)

    return x, y, vx, vy


def create_simulations():
    for i in range(NUM_SIMULATIONS):
        x, y, vx, vy = initialize_arrays()
        x, y, vx, vy = verlet_ode_solver(x, y, vx, vy, i)
        print_planets(x, y, i)
        x, y, vx, vy = normalize_simulation(x, y, vx, vy, i)


def create_start_states():
    start_states = np.zeros((1, 12))
    for i in range(LOCKBOX_SIZE):
        x, y, vx, vy = initialize_arrays()
        start_state = np.hstack((x, y, vx, vy))
        start_states = np.vstack((start_states, start_state[0]))
    start_states = np.delete(start_states, 0, axis=0)

    df = pd.DataFrame(start_states)
    df.to_csv(os.getcwd() + "/data/lockbox/start_states.csv", header=False, index=False)


def main():
    # NEW CODE uses velocities as initial conditions - based on Li, Liao
    create_simulations()
    create_start_states()


if __name__ == "__main__":
    main()
