"""
Author: Boris de Jong
"""


"""...........................PACKAGES..........................."""

import numpy as np, matplotlib.pyplot as plt, os
dir = os.path.dirname(__file__)
from scipy.optimize import fsolve

"""...........................CONSTANTS.........................."""

c = 3e8                             # speed of light [m/s]
lambda0 = 300e-9                    # wavelength [m]
omega = 2 * np.pi * c / lambda0     # angular frequency [rad/s]
k0 = omega / c                      # free space wavevector
epsilon_m = -10 + 1j                # metal
epsilon_d = 2 + 0j               # dielectric

"""...........................FUNCTIONS.........................."""

beta = k0 * np.sqrt(epsilon_m * epsilon_d / (epsilon_m + epsilon_d) + 0j)  # propagation constant
neff = beta / k0  # effective index
k1 = np.sqrt(beta**2 - epsilon_m * k0**2 + 0j)
k2 = np.sqrt(beta**2 - epsilon_d * k0**2 + 0j)

"""...........................NUMERICAL SOLUTION................"""

def dispersion_complex(beta_vec):
    beta = beta_vec[0] + 1j * beta_vec[1]
    k1 = np.sqrt(beta**2 - epsilon_m * k0**2 + 0j)
    k2 = np.sqrt(beta**2 - epsilon_d * k0**2 + 0j)
    return [np.real(k1 / epsilon_m + k2 / epsilon_d), np.imag(k1 / epsilon_m + k2 / epsilon_d)]

beta_initial_guess = [np.real(k0 * np.sqrt(epsilon_d)), 1e4]
beta_solution = fsolve(dispersion_complex, beta_initial_guess)
beta_solution_real, beta_solution_imag = beta_solution
beta_solution = beta_solution_real + 1j * beta_solution_imag

k1_num = np.sqrt(beta_solution**2 - epsilon_m * k0**2)
k2_num = np.sqrt(beta_solution**2 - epsilon_d * k0**2)

""".............................MAIN............................."""

z = np.linspace(-400e-9, 400e-9, 500)
# interface at z = 0 nm
Hy_m_num = np.exp(k1_num * z[z < 0])    # decay in metal
Hy_d_num = np.exp(-k2_num * z[z >= 0])  # decay in dielectric
Hy_m = np.exp(k1 * z[z < 0])            # decay in metal
Hy_d = np.exp(-k2 * z[z >= 0])          # decay in dielectric
Hy_m_num /= np.max(Hy_m_num)
Hy_d_num /= np.max(Hy_d_num)
Hy_m /= np.max(Hy_m)
Hy_d /= np.max(Hy_d)

plt.plot(z[z < 0] * 1e9, Hy_m, label='metal region')
plt.plot(z[z >= 0] * 1e9, Hy_d, label='dielectric region')
plt.plot(z[z < 0] * 1e9, Hy_m_num, label='metal region num', linestyle='--')
plt.plot(z[z >= 0] * 1e9, Hy_d_num, label='dielectric region num', linestyle='--')
plt.axvline(0, color='black', linestyle=':', label='interface')
plt.xlabel(r'$z$ (nm)')
plt.ylabel(r'magnetic field $H_y$ (norm.)')
plt.legend()
plt.show(); plt.close()
