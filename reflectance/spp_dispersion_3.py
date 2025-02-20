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

"""..........................MCPEAK DATA........................."""

f = os.path.join(dir, "refractive_index_Ag.csv") # McPeak refractiveindex.info
d = pd.read_csv(f)
wavelengths = d['wl'].values[:49].astype(float)
n = d['n'].values[:49].astype(float)
k = d['n'].values[50:].astype(float)

re_eps = n**2 - k**2 # real part of permittivity
im_eps = 2 * k * n # imaginary part of permittivity

lamb_arr = np.linspace(0.400, 1.200, 1000)  # lambda [μm]
eps_m = np.interp(lamb_arr, wavelengths, re_eps + 1j * im_eps)  # metal permittivity interpolated for values in lamb_arr
eps_d = 1.00  # vacuum permittivity

k0 = (2 * np.pi) / (lamb_arr * 1e-6)  # angular wave number [rad/m]
omega = c * k0  # angular frequency [rad/s]

n_eff = np.sqrt((eps_d * eps_m) / (eps_d + eps_m))  # effective index
k_spp = k0 * n_eff  # SPP wave number [rad/m]
E_vals = (hbar * omega) / e  # energy in [eV]

"""...........................FUNCTIONS.........................."""


def compute_eig(k: float) -> tuple: 
    """
    Computes the eigenvalues and eigenvectors of the coupling matrices for a given wavevector k.

    The eigenvalues represent the energy levels, while the eigenvectors provide 
    information on the mode composition.

    Parameters:
        k (float): The in-plane wavevector component (in m⁻¹).

    Returns:
        tuple: 
            - eigvals1 (np.ndarray): Eigenvalues of the first coupling matrix (energy levels in eV).
            - eigvecs1 (np.ndarray): Corresponding eigenvectors of the first coupling matrix.
            - eigvals2 (np.ndarray): Eigenvalues of the second coupling matrix (energy levels in eV).
            - eigvecs2 (np.ndarray): Corresponding eigenvectors of the second coupling matrix.
    """
    eigvals, eigvecs = eig(np.array([
        [E_of_k_right(k), gamma_eV],
        [gamma_eV, E_of_k_left(k)]
    ]))

    return eigvals, eigvecs

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generates a 1D Gaussian kernel (for convolution)

    Parameters:
        size (int): The size of the kernel (number of points).
        sigma (float): The standard deviation of the Gaussian function.

    Returns:
        np.ndarray: A 1D Gaussian kernel normalized so that its sum equals 1.
    """
    x = np.linspace(-size // 2, size // 2, size)
    gauss = np.exp(-0.5 * (x / sigma) ** 2)

    return gauss / gauss.sum()

""".........................CALCULATION........................"""

# initialize interpolation functions E(k)
n = 1 # interpolation of E(k±ng)
E_of_k_right = interp1d(np.real(k_spp - (n * g)), np.real(E_vals), kind='linear', fill_value="extrapolate")
E_of_k_left = interp1d(np.real(-k_spp + (n * g)), np.real(E_vals), kind='linear', fill_value="extrapolate")

k_vals = np.linspace(-7e6, 7e6, 1000) 

# plt.plot(k_vals, E_of_k_right(k_vals))
# plt.plot(k_vals, E_of_k_left(k_vals))
# plt.show()

evector_arr =  np.zeros(shape=k_vals.shape, dtype=object); eval_arr = np.zeros(shape=k_vals.shape, dtype=object)

for i, k in enumerate(k_vals):
    eval, evec = compute_eig(k)
    evector_arr[i] = evec
    eval_arr[i] = eval

eigenvectors_separated = [[evec[:, j] for j in range(evec.shape[1])] for evec in evector_arr]


print(eigenvectors_separated[0:, 0])
