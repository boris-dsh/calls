"""
Author: Boris de Jong
"""


"""...........................PACKAGES..........................."""

import numpy as np, matplotlib.pyplot as plt, os
dir = os.path.dirname(__file__)

"""...........................CONSTANTS.........................."""

L = 1
kappa = 1
delta_L_vals = np.linspace(-15, 15, 200)
alpha_L_vals = np.linspace(0, 10, 200)
Delta, Alpha = np.meshgrid(delta_L_vals, alpha_L_vals)
Eq1_vals = np.zeros_like(Delta)
Eq2_vals = np.zeros_like(Delta)
Eq3_vals = np.zeros_like(Delta)
Eq4_vals = np.zeros_like(Delta)

"""...........................FUNCTIONS.........................."""



""".............................MAIN............................."""

for i in range(len(delta_L_vals)):
    for j in range(len(alpha_L_vals)):
        delta_L, alpha_L = delta_L_vals[i], alpha_L_vals[j]
        gamma = np.sqrt(kappa**2 + (alpha_L - 1j * delta_L)**2)
        sinh_gamma_L = np.sinh(gamma * L)
        coth_gamma_L = np.cosh(gamma * L) / sinh_gamma_L
        
        Eq1_vals[j, i] = np.imag(gamma / sinh_gamma_L) - kappa
        Eq2_vals[j, i] = np.real(gamma / sinh_gamma_L)
        Eq3_vals[j, i] = np.real(gamma * coth_gamma_L) - alpha_L
        Eq4_vals[j, i] = -np.imag(gamma * coth_gamma_L) - delta_L

plt.figure(figsize=(5, 5))
plt.contour(Delta, Alpha, Eq1_vals, levels=[0], colors='C0')
plt.contour(Delta, Alpha, Eq1_vals, levels=[0], colors='C0', linestyle=':')
plt.contour(Delta, Alpha, Eq2_vals, levels=[0], colors='C1')
plt.contour(Delta, Alpha, Eq3_vals, levels=[0], colors='C2')
plt.contour(Delta, Alpha, Eq4_vals, levels=[0], colors='C3')

plt.xlabel(r'$\delta L$')
plt.ylabel(r'$\alpha L$')
plt.show();plt.close()



