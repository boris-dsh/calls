"""
Author: Boris de Jong
"""


"""...........................PACKAGES..........................."""

import numpy as np, matplotlib.pyplot as plt, os, pandas as pd
from scipy.interpolate import interp1d
from numpy.linalg import eig 
from scipy.signal import fftconvolve
dir = os.path.dirname(__file__)

"""...........................CONSTANTS.........................."""

c = 2.998e8  # light speed [m/s]
h = 6.62607015e-34  # planck's constant [Js]
hbar = h / (2 * np.pi)  # reduced planck's constant [Js]
e = 1.602176634e-19  # elementary charge [C]
gamma = 0.05 * e  # coupling energy [J]
gamma_eV = gamma / e # coupling energy [eV]
p = 600e-9  # pitch size of grating [m]
g = 2 * np.pi / p  # reciprocal wave vector [rad/m]
L = 5e-6 # grating size [m]
phi = np.pi / 2  # phase shift
x_vals = np.linspace(-L/2, L/2, 2000) # spatial width of grating 
h_prime_x = -g * np.sin(g * x_vals + phi) # assume cosine form of first harmonic, so derivative -> -sin
plasma_freq = 9.01e15 # plasma frequency of silver [Hz]
collision_freq = 1.38e14 # collision frequency of silver [Hz]

"""..........................FUNCTIONS........................."""

def n_eff_func(omega: float) -> complex:
    """
    Computes the effective index for a given angular frequency omega.
    """
    eps_m = 1 - (plasma_freq**2) / (omega**2 + 1j * collision_freq * omega)  # metal permittivity
    eps_d = 1.00  # vacuum permittivity
    return np.sqrt((eps_d * eps_m) / (eps_d + eps_m))  # effective index

def k_spp(k: list) -> np.ndarray:
    """
    Computes the SPP wave number for a given angular wave number k0.
    """
    k = np.asarray(k)
    omega = c * k  # angular frequency [rad/s]
    neff = n_eff_func(omega)
    return k * neff  # SPP wave number [rad/m]

def k_spp_func(omega: float) -> float:
    """
    Computes the SPP wave number for a given angular frequency omega.
    """
    n_eff = n_eff_func(omega)
    return (omega / c) * n_eff

def e_field(k: float) -> np.ndarray:
    """
    Computes the electric field for a given angular wave number k.
    """
    if k == 0:
        kspp = 0
        n_eff = 1
    else:
        kspp = k_spp(k)
        omega = c * k  # angular frequency [rad/s]
        n_eff = n_eff_func(omega)  # effective index
    norm_func_plus = (np.sqrt((hbar * c * kspp)**2 + (gamma * n_eff)**2) + (hbar * c * kspp)) / (gamma * n_eff)
    norm_func_minus = (np.sqrt((hbar * c * kspp)**2 + (gamma * n_eff)**2) - (hbar * c * kspp)) / (gamma * n_eff)

    e_field_plus = 1/np.sqrt(1 + norm_func_plus**2) * (np.exp(1j * (kspp + g) * x_vals) + norm_func_plus * np.exp(1j * (kspp - g) * x_vals))  # electric field +
    e_field_minus = 1/np.sqrt(1 + norm_func_minus**2) * (np.exp(1j * (kspp + g) * x_vals) - norm_func_minus * np.exp(1j * (kspp - g) * x_vals))  # electric field -
    return e_field_plus, e_field_minus # electric fields

def energy_kspp_interpolation():
    """
    Creates an interpolation function for energy as a function of k_spp.
    """
    omega_vals = np.linspace(1e5 * c, 1e7 * c, 1000)  # range of angular frequencies
    k_spp_vals = np.array([k_spp_func(omega).real for omega in omega_vals])
    energy_vals = (hbar * omega_vals / e).real  # energy in [eV]

    return interp1d(k_spp_vals-g, energy_vals, kind='linear', fill_value="extrapolate"), interp1d(-k_spp_vals+g, energy_vals, kind='linear', fill_value="extrapolate")

def eigenvalues(k: float) -> np.ndarray:
    """
    Computes the eigenvalues of the coupling matrices for a given wavevector k.
    """
    A = np.array([[energy_from_kspp_minus(k), gamma_eV], [gamma_eV, energy_from_kspp_plus(k)]])
    return np.linalg.eigvals(A)  # eigenvalues

"""............................MAIN..........................."""

energy_from_kspp_minus, energy_from_kspp_plus = energy_kspp_interpolation()

kv = np.linspace(-1e7, 1e7, 1000)
energy_values = energy_from_kspp_minus(kv)

plt.plot(kv, energy_from_kspp_minus(kv), label='E(kspp-)')
plt.plot(kv, energy_from_kspp_plus(kv), label='E(kspp)+')
plt.xlabel('k_spp [rad/m]')
plt.ylabel('energy [eV]')
plt.legend()
plt.show(); plt.close()

eigenvalues_array = np.array([eigenvalues(k) for k in kv])
plt.plot(kv, eigenvalues_array[:, 0], 'o', label='E1')
plt.plot(kv, eigenvalues_array[:, 1], 'o', label='E2')
plt.xlabel('k [rad/m]')
plt.ylabel('eigenvalues [eV]')
plt.legend()
plt.show(); plt.close()

plt.plot(x_vals, (e_field(0)[0]), label='E+') # at k0 = 0: ~cos(gx)
plt.plot(x_vals, np.imag(e_field(0)[1]), label='E-') # at k0 = 0: ~sin(gx)
plt.legend()
plt.show(); plt.close()

intensity_plus = []
intensity_minus = []

for i in range(len(kv)):
    intensity_plus.append(np.trapezoid(h_prime_x * (e_field(kv[i])[0]) * np.exp(1j * kv[i] * x_vals), x_vals))
    intensity_minus.append(np.trapezoid(h_prime_x * ((e_field(kv[i])[1])) * np.exp(1j * kv[i] * x_vals), x_vals))
#     # if i % 50 == 0:
#     #     plt.plot(x_vals, h_prime_x * (e_field(kv[i])[0]) * np.exp(1j * kv[i] * x_vals), label='h\'(x) * E+')
#     #     plt.plot(x_vals, h_prime_x * (np.imag(e_field(kv[i])[1])) * np.exp(1j * kv[i] * x_vals), label='h\'(x) * E-')
#     #     plt.legend()
#     #     plt.show(); plt.close()

plt.plot(kv, np.abs(intensity_plus)**2, label='int +')
plt.plot(kv, np.abs(intensity_minus)**2, label='int -')
plt.legend()
plt.show(); plt.close()

energy_vals = np.linspace(energy_from_kspp_minus(-1e7), energy_from_kspp_minus(1e7), 1000)
intensity_grid = np.zeros((len(energy_vals), len(kv)))

for i, k in enumerate(kv):
    energy_minus = energy_from_kspp_minus(k)
    energy_plus = energy_from_kspp_plus(k)
    energy_idx = np.abs(energy_vals - eigenvalues(k)[0]).argmin()
    intensity_grid[energy_idx, i] = np.abs(intensity_plus[i])**2
    energy_idx = np.abs(energy_vals - eigenvalues(k)[1]).argmin()
    intensity_grid[energy_idx, i] = np.abs(intensity_minus[i])**2

plt.imshow(intensity_grid, extent=[kv[0], kv[-1], energy_vals[0], energy_vals[-1]], aspect='auto', origin='lower', cmap='plasma')
plt.colorbar(label='intensity')
plt.xlabel('k [rad/m]')
plt.ylabel('energy [eV]')
plt.show(); plt.close()


plt.plot(kv, intensity_grid.sum(axis=0))
plt.xlabel('k [rad/m]')
plt.ylabel('intensity')
plt.show(); plt.close()