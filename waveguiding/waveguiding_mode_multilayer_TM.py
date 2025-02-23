"""
Author: Boris de Jong
"""


"""...........................PACKAGES..........................."""

import numpy as np, matplotlib.pyplot as plt, os
from scipy.optimize import minimize_scalar, root
dir = os.path.dirname(__file__)

"""...........................CONSTANTS.........................."""

epsilon_0 = 8.854187817e-12     # permittivity of free space [F/m]
c = 3e8                         # speed of light [m/s]
a = 1e-6                        # core thickness [m]
epsilon_1 = 2.25                # rel. permittivity of core
epsilon_2 = 1.00                # rel. permittivity of cladding 2
epsilon_3 = -10 + 1j            # rek. permittivity of cladding 3
lambda0 = 1.55e-6               # wavelength [m]
omega = 2 * np.pi * c / lambda0 # angular frequency [rad/s]
k0 = omega / c                  # free space wavevector

"""...........................FUNCTIONS.........................."""

def dispersion_relation(beta):

    k1 = np.sqrt(beta**2 - k0**2 * epsilon_1 + 0j)
    k2 = np.sqrt(beta**2 - k0**2 * epsilon_2 + 0j)
    k3 = np.sqrt(beta**2 - k0**2 * epsilon_3 + 0j)
    
    numerator = (k1 / epsilon_1 + k2 / epsilon_2) * (k1 / epsilon_1 + k3 / epsilon_3)
    denominator = (k1 / epsilon_1 - k2 / epsilon_2) * (k1 / epsilon_1 - k3 / epsilon_3)
    
    return np.exp(-4 * k1 * a) - numerator / denominator


""".............................MAIN............................."""


# plt.plot(np.linspace(0, 1e6, 1000), [dispersion_relation(beta) for beta in np.linspace(0, 1e6, 1000)])
# plt.xlabel(r'$\beta$')
# plt.ylabel(r'$f(\beta)$')
# plt.show(); plt.close()

beta_min = k0 * np.sqrt(np.min([epsilon_2, epsilon_3]) + 0j)
beta_max = k0 * np.sqrt(epsilon_1 + 0j)
beta = minimize_scalar(dispersion_relation, bounds=(beta_min, beta_max), method='bounded')

if beta.success:
    beta_solution = beta.x
    print(f"Solved propagation constant beta: {beta_solution}")
else:
    print("Failed to solve for beta.")
    exit()

k1 = np.sqrt(beta_solution**2 - k0**2 * epsilon_1)
k2 = np.sqrt(beta_solution**2 - k0**2 * epsilon_2)
k3 = np.sqrt(beta_solution**2 - k0**2 * epsilon_3)

print(f"k1: {k1}, k2: {k2}, k3: {k3}")

def equations(variables):
    eq1 = variables[0] * np.exp(-k3 * a) - (variables[2] * np.exp(k1 * a) + variables[3] * np.exp(-k1 * a))
    eq2 = (variables[0] / epsilon_3) * k3 * np.exp(-k3 * a) - (-(variables[2] / epsilon_1) * k1 * np.exp(k1 * a) + (variables[3] / epsilon_1) * k1 * np.exp(-k1 * a))
    eq3 = variables[1] * np.exp(-k2 * a) - (variables[2] * np.exp(-k1 * a) + variables[3] * np.exp(k1 * a))
    eq4 = -(variables[1] / epsilon_2) * k2 * np.exp(-k2 * a) - (-(variables[2] / epsilon_1) * k1 * np.exp(-k1 * a) + (variables[3] / epsilon_1) * k1 * np.exp(k1 * a))
    return [eq1, eq2, eq3, eq4]

initial_guesses = [2000, 168, 1, 1]
# solution = root(equations, initial_guesses)


# if solution.success:
#     A, B, C, D = solution.x
#     print(f"Constants: A = {A}, B = {B}, C = {C}, D = {D}")
# else:
#     print("Failed to solve for A, B, C, D.")
#     exit()

A, B, C, D = initial_guesses

z = np.linspace(-2*a, 2*a, 1000)
Hy = np.zeros_like(z)
Ex = np.zeros_like(z)
Ez = np.zeros_like(z)

for i, zi in enumerate(z):
    if zi < -a:
        Hy[i] = B * np.exp(k2 * zi)
        Ex[i] = -1j * B * (k2 / (omega * epsilon_0 * epsilon_2)) * np.exp(k2 * zi)
        Ez[i] = -B * (beta_solution / (omega * epsilon_0 * epsilon_2)) * np.exp(k2 * zi)
    elif -a <= zi <= a:
        Hy[i] = C * np.exp(k1 * zi) + D * np.exp(-k1 * zi)
        Ex[i] = -1j * C * (k1 / (omega * epsilon_0 * epsilon_1)) * np.exp(k1 * zi) + 1j * D * (k1 / (omega * epsilon_0 * epsilon_1)) * np.exp(-k1 * zi)
        Ez[i] = C * (beta_solution / (omega * epsilon_0 * epsilon_1)) * np.exp(k1 * zi) + D * (beta_solution / (omega * epsilon_0 * epsilon_1)) * np.exp(-k1 * zi)
    else:
        Hy[i] = A * np.exp(-k3 * zi)
        Ex[i] = -1j * A * (k3 / (omega * epsilon_0 * epsilon_3)) * np.exp(-k3 * zi)
        Ez[i] = -A * (beta_solution / (omega * epsilon_0 * epsilon_3)) * np.exp(-k3 * zi)

plt.figure()
plt.plot(z, np.real(Hy), label='Hy')
# plt.plot(z, np.real(Ex), label='Ex')
# plt.plot(z, np.real(Ez), label='Ez')
plt.axvline(x=-a, color='black', linestyle='--', label='interface 1')
plt.axvline(x=a, color='black', linestyle='--', label='interface 2')
plt.xlabel(r'$z$ (m)')
plt.ylabel('Field Amplitude')
plt.legend()
plt.show(); plt.close()