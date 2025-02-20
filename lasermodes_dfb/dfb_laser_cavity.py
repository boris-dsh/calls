"""
Author: Boris de Jong
"""


"""...........................PACKAGES..........................."""

import numpy as np, matplotlib.pyplot as plt, os
dir = os.path.dirname(__file__)

"""...........................CONSTANTS.........................."""

L = 1.0
N = 1000
dx = L / N
alpha = [1.75, 5.95, 9.16, 12.35]
delta = [2.66, 2.52, 2.93, 3.23]
kappa = 6
round_trips = 2
r1 = 0.9
r2 = 0.9
mode = 1

"""...........................FUNCTIONS.........................."""



""".............................MAIN............................."""

R = np.zeros(N, dtype=complex)
S = np.zeros(N, dtype=complex)
R[0] = 1 # start intensity on left-hand side of cavity

R_intensity = np.zeros((round_trips, N))
S_intensity = np.zeros((round_trips, N))
tot_int = np.zeros((round_trips, N))

for trip in range(round_trips): # approximate derivative dR/dx [dS/dx] by (R[j] - R[j-1])/dx [S[j] - S[j-1])/dx]
    for j in range(1, N): # left to right (increasing j)
        R[j] = R[j-1] + dx * (alpha[mode] - 1j*delta[mode]) * R[j-1] - 1j*kappa * S[j-1]
    for j in range(N-2, -1, -1): # right lo left (decreasing j)
        S[j] = S[j+1] - dx * (alpha[mode] - 1j*delta[mode]) * S[j+1] - 1j*kappa * R[j+1]

    R[0] = r1 * S[0]
    S[-1] = r2 * R[-1]

    R_intensity[trip, :] = np.abs(R)**2
    S_intensity[trip, :] = np.abs(S)**2
    tot_int[trip, :] = np.abs(R)**2 + np.abs(S)**2

""".............................PLOT............................."""

x = np.linspace(0, L, N)
plt.figure(figsize=(6, 4))
plt.plot(x, tot_int[trip, :])
plt.xlabel(r'$x$')
plt.ylabel(r"$|R|^2$ + $|S|^2$")
plt.show()

for trip in range(round_trips):
    plt.plot(x, R_intensity[trip, :], label=f"R {trip+1}")
    plt.plot(x, S_intensity[trip, :], label=f"S {trip+1}", linestyle=":")

plt.yscale('log')
plt.xlabel(r'$x$')
plt.ylabel(r"$|R|^2$, $|S|^2$")
plt.show(); plt.close()