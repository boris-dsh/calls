"""
Author: Boris de Jong
"""

"""...........................PACKAGES..........................."""

import numpy as np, matplotlib.pyplot as plt, os
import scipy.optimize as opt
dir = os.path.dirname(__file__)

"""...........................CONSTANTS.........................."""

epsilon_1 = -15.0 + 1.0j
epsilon_2 = 2.25
epsilon_3 = 1.0
d = 400e-9
wavelength = 700e-9
k0 = 2 * np.pi / wavelength

"""...........................FUNCTIONS.........................."""

def dispersion_relation(beta, epsilon_1, epsilon_2, epsilon_3, d, k0):
    kz1 = np.sqrt(epsilon_1 * k0**2 - beta**2)
    kz2 = np.sqrt(epsilon_2 * k0**2 - beta**2)
    kz3 = np.sqrt(epsilon_3 * k0**2 - beta**2)

    M2 = np.array([
        [np.cos(kz2 * d), (epsilon_2 / (kz2 * epsilon_1)) * np.sin(kz2 * d)],
        [-kz2 * np.sin(kz2 * d), (epsilon_2 / epsilon_1) * np.cos(kz2 * d)]
    ])

    T3 = np.array([
        [1, 0],
        [0, epsilon_3 / epsilon_2]
    ])
    
    Q3 = T3 @ M2
    
    # dispersion relation
    return Q3[0, 0] - 1j * kz1 * Q3[0, 1] + (1j / kz3) * Q3[1, 0] + (kz1 / kz3) * Q3[1, 1]

""".............................MAIN............................."""

def dispersion_relation_wrapper(beta, epsilon_1, epsilon_2, epsilon_3, d, k0):
    beta_complex = beta[0] + 1j * beta[1]
    result = dispersion_relation(beta_complex, epsilon_1, epsilon_2, epsilon_3, d, k0)
    return np.array([np.real(result), np.imag(result)])

beta_guess_real = 1.5 * k0
beta_guess_imag = 0.1 * k0
beta_guess = np.array([beta_guess_real, beta_guess_imag])

result = opt.root(dispersion_relation_wrapper, beta_guess, args=(epsilon_1, epsilon_2, epsilon_3, d, k0), method='lm')

beta_solution = result.x[0] + 1j * result.x[1]
print(f"Found beta: {beta_solution}")
n_eff = beta_solution / k0
print(f"Effective index: {n_eff}")

kz1 = np.sqrt(epsilon_1 * k0**2 - beta_solution**2)
kz2 = np.sqrt(epsilon_2 * k0**2 - beta_solution**2)
kz3 = np.sqrt(epsilon_3 * k0**2 - beta_solution**2)

M2 = np.array([
    [np.cos(kz2 * d), (epsilon_2 / (kz2 * epsilon_1)) * np.sin(kz2 * d)],
    [-kz2 * np.sin(kz2 * d), (epsilon_2 / epsilon_1) * np.cos(kz2 * d)]
])

T3 = np.array([
    [1, 0],
    [0, epsilon_3 / epsilon_2]
])

Q3 = T3 @ M2

def calculate_mode_profile(beta, epsilon_1, epsilon_2, epsilon_3, d, k0, Q3):
    kz1 = np.sqrt(epsilon_1 * k0**2 - beta**2)
    kz2 = np.sqrt(epsilon_2 * k0**2 - beta**2)
    kz3 = np.sqrt(epsilon_3 * k0**2 - beta**2)
    
    z = np.linspace(-d, d, 1000)
    Hy = np.zeros_like(z, dtype=complex)
    
    for i, zi in enumerate(z):
        if zi < -d/2:
            Hy[i] = np.exp(-1j * kz1 * (zi))
        elif -d/2 <= zi <= d/2:
            Hy[i] = (Q3[0, 0] + 1j * kz1 * Q3[0, 1]) * np.cos(kz2 * (zi - d/2)) + (Q3[1, 0] + 1j * kz1 * Q3[1, 1]) * (1 / kz2) * 2*np.sin(kz2 * (zi - d/2))
        else:
            Hy[i] = (Q3[0, 0] + 1j * kz1 * Q3[0, 1]) * np.exp(-1j * kz3 * (zi))
    
    return z, Hy

z, Hy = calculate_mode_profile(beta_solution, epsilon_1, epsilon_2, epsilon_3, d, k0, Q3)

plt.plot(z * 1e9, np.real(Hy))
plt.xlabel('z (nm)')
plt.ylabel('H_y(z)')
plt.show()
