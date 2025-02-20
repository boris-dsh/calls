"""
Author: Boris de Jong
"""

"""...........................PACKAGES..........................."""

import numpy as np, matplotlib.pyplot as plt, os
from scipy.signal import fftconvolve
dir = os.path.dirname(__file__)

"""...........................FUNCTIONS.........................."""

def lambda1(k):
    return (hbar * c * g + np.sqrt((hbar * c * k)**2 + (b * n)**2)) / n

def lambda2(k):
    return (hbar * c * g - np.sqrt((hbar * c * k)**2 + (b * n)**2)) / n

def k1(E):
    arg = (E * n - hbar * c * g)**2 - (b * n)**2
    return (1 / (hbar * c)) * np.sqrt(np.sqrt(arg**2))

def k2(E):
    arg = (E * n - hbar * c * g)**2 - (b * n)**2
    return -(1 / (hbar * c)) * np.sqrt(np.sqrt(arg**2))

def wavelength_to_energy(wavelength_nm):
    wavelength_m = wavelength_nm * 1e-9
    energy_J = (2 * np.pi * hbar * c) / wavelength_m
    energy_eV = energy_J / e
    return energy_eV

"""...........................CONSTANTS.........................."""

hbar = 6.62607015e-34 / (2 * np.pi) # [Js]
c = 2.99792458e8 # [m/s]
e = 1.602176634e-19 # [C]
l = 5e-6 # size of grating [m]
p = 600e-9 # pitch size of grating [m]
g = 2 * np.pi / p # reciprocal wave vector [1/m]
n = 1.03 # effective mode index
b = 0.06 * e # coupling energy [J]

""".............................MAIN............................."""

max_angle = np.arcsin(0.8 / 1) * (180 / np.pi)  # NA = 0.8, n = 1.0 for air
angles = np.linspace(-max_angle, max_angle, 1000)
wavelength_grid, angle_grid = np.meshgrid(np.linspace(400, 700, 1000), angles, indexing='ij')
k_values = 1e9 * (2 * np.pi * np.sin(np.radians(angle_grid))) / wavelength_grid # 1/m

# k_values = np.linspace(-10, 10, 1500) * 1e6 # [1/m]
energy_min, energy_max = lambda2(k_values).min() / e, lambda1(k_values).max() / e # [eV]
# print(k_values)
energy_grid = wavelength_to_energy(wavelength_grid)
energy_values = np.linspace(energy_min, energy_max, 1000) # [eV]
# _, E = np.meshgrid(angles, energy_values)

intensity = np.zeros_like(k_values)
intensity[np.abs(energy_grid - lambda1(k_values) / e) < 3e-3] = 1
intensity[np.abs(energy_grid - lambda2(k_values) / e) < 3e-3] = 1

""".............................CONVOLUTION............................."""

sinc_kernel = np.sinc(k_values[0] * l / (2))
sinc_kernel /= np.trapezoid(sinc_kernel, k_values[0]) # normalize sinc kernel
intensity_convolved = np.array([
    fftconvolve(intensity[i], np.abs(sinc_kernel), mode='same')
    for i in range(intensity.shape[0])
]) # convolution of the intensity matrix with the sinc kernel each row

""".............................PLOTTING............................."""

plt.figure(figsize=(4,6))
ax1 = plt.gca() # get current axis
pcm = ax1.pcolormesh(k_values * 1e-6, energy_grid, 1/(intensity_convolved+0.001), cmap='viridis')
plt.colorbar(pcm, ax=ax1, label='intensity', location='top')
ax1.set_xlabel('$k_x$ (rad/$\mu$m)')
ax1.set_ylabel('energy (eV)')

ax2 = ax1.twinx() # same x-axis
ymin, ymax = ax1.get_ylim()
ax2.set_ylim(ymin, ymax)

custom_wavelength_ticks = np.array([400, 500, 600, 700])
custom_energy_ticks = wavelength_to_energy(custom_wavelength_ticks) # wavelength to eV

ax2.set_yticks(custom_energy_ticks)
ax2.set_yticklabels([f'{w}' for w in custom_wavelength_ticks])
ax2.set_ylabel('wavelength (nm)')
plt.tight_layout()

# plt.savefig(
plt.show()

