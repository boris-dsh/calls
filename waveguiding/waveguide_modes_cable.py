"""
Author: Boris de Jong
"""

"""...........................PACKAGES..........................."""

import numpy as np, matplotlib.pyplot as plt, os
from scipy.optimize import fsolve
dir = os.path.dirname(__file__)

"""...........................CONSTANTS.........................."""


"""...........................FUNCTIONS.........................."""

def dispersion_relation(kx, k0, n1, n2, L):
    kz1 = np.sqrt(k0**2 * n1**2 - kx**2)
    gamma = np.sqrt(kx**2 - k0**2 * n2**2)
    return np.tan(kz1 * L / 2) - (gamma / kz1)

def solve_kx(k0, n1, n2, L):
    kx_guess = k0 * (n1 + n2) / 2
    kx_solution = fsolve(dispersion_relation, kx_guess, args=(k0, n1, n2, L))
    return kx_solution[0]

def mode_profile(z, kx, k0, n1, n2, L):
    kz1 = np.sqrt(k0**2 * n1**2 - kx**2)
    gamma = np.sqrt(kx**2 - k0**2 * n2**2)
    
    A1 = 1
    A2 = A1 * np.cos(kz1 * L / 2) / np.exp(-gamma * L / 2)
    
    phi = np.piecewise(z, 
                        [abs(z) <= L/2, abs(z) > L/2],
                        [lambda z: A1 * np.cos(kz1 * z),
                         lambda z: A2 * np.exp(-gamma * np.abs(z))])
    return phi

def plot_mode(L_values):
    k0 = 2 * np.pi / 1500
    n1, n2 = 1.5, 1.33
    
    z = np.linspace(-10000, 10000, 5000)
    plt.figure(figsize=(10, 6))
    
    for L in L_values:
        kx = solve_kx(k0, n1, n2, L)
        phi = mode_profile(z, kx, k0, n1, n2, L)
        plt.plot(z, phi)

    plt.show()
    

""".............................MAIN............................."""


plot_mode([10000, 1000, 200])
