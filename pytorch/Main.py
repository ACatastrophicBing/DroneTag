import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def quadrotor(t, z, u, p, r, n):
    # Unpacking the parameters
    g, l, m, I11, I22, I33, mu, sigma = p

    # Forming the moment of inertia tensor based on the parameter values
    I = np.diag([I11, I22, I33])

    # Rotation matrix mapping body fixed frame C to inertial frame E
    R = np.array([
        [np.cos(z[4]) * np.cos(z[5]), np.sin(z[3]) * np.sin(z[4]) * np.cos(z[5]) - np.cos(z[3]) * np.sin(z[5]),
         np.sin(z[3]) * np.sin(z[5]) + np.cos(z[3]) * np.sin(z[4]) * np.cos(z[5])],
        [np.cos(z[4]) * np.sin(z[5]), np.cos(z[3]) * np.cos(z[5]) + np.sin(z[3]) * np.sin(z[4]) * np.sin(z[5]),
         np.cos(z[3]) * np.sin(z[4]) * np.sin(z[5]) - np.sin(z[3]) * np.cos(z[5])],
        [-np.sin(z[4]), np.sin(z[3]) * np.cos(z[4]), np.cos(z[3]) * np.cos(z[4])]
    ])

    # Adjusting thrust output based on feasible limits
    nu = np.clip(u, 0, mu)

    # Computing temporary variables
    rt = np.array([(u[1] - u[3]) * l, (u[2] - u[0]) * l, (u[0] - u[1] + u[2] - u[3]) * sigma])

    # Computing time derivative of the state vector
    dz = np.zeros(12)
    dz[0:3] = z[6:9]
    dz[3:6] = [
        z[9] + z[11] * np.cos(z[3]) * np.tan(z[4]) + z[10] * np.sin(z[3]) * np.tan(z[4]),
        z[10] * np.cos(z[3]) - z[11] * np.sin(z[3]),
        (z[11] * np.cos(z[3]) + z[10] * np.sin(z[3])) / np.cos(z[4])
    ]
    dz[6:9] = np.dot(R, (np.array([0, 0, sum(u)]) + r) / m) - np.array([0, 0, g])
    dz[9:12] = np.linalg.solve(I, rt + n - np.cross(z[9:12], np.dot(I, z[9:12])))

    return dz

# Parameters and initializations
g = 9.81  # gravitational acceleration [m/s^2]
l = 0.2  # distance from the center of mass to each rotor [m]
m = 0.5  # total mass of the quadrotor [kg]
I = [1.24, 1.24, 2.48]  # mass moment of inertia [kg m^2]
mu = 3.0  # maximum thrust of each rotor [N]
z0 = np.zeros([12,1],1)

# ... (you might want to define the control inputs, disturbances, and initial states here)

# Integration using odeint
times = np.linspace(0, 10, 1000)  # you can adjust the time span and steps
states = odeint(quadrotor, z0, times, args=(u, p, r, n))


# ... (assuming states and times are already defined)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 4])

# Initial drawing
quadrotor_draw, = ax.plot([], [], [], 'b-')  # adjust the style as needed


def init():
    quadrotor_draw.set_data([], [])
    quadrotor_draw.set_3d_properties([])
    return quadrotor_draw,


def update(frame):
    # ... (code to calculate the quadrotor's position and orientation)
    quadrotor_draw.set_data(x, y)  # x, y are the calculated positions
    quadrotor_draw.set_3d_properties(z)  # z is the calculated height
    return quadrotor_draw,


ani = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=True)

plt.show()

exp_done = False
episode_done = False
num_episodes = 1000

while not exp_done:
    # Reset the environment, make changes to environment / different starting positions / etc
    while not episode_done:
        time_steps = 0
         # Get robot observations from environment
         # Collect rewards for being in the environment
         # Do some fancy stuff that I don't want to handle right now

