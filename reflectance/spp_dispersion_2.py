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

k_vals = np.linspace(-7e6, 7e6, 500) 

plt.plot(k_vals, E_of_k_right(k_vals))
plt.plot(k_vals, E_of_k_left(k_vals))
plt.show()

energy_vals = np.linspace(1.8, 3, 1000)
intensity_map = np.zeros((len(energy_vals), len(k_vals)))


M1_list = []; M2_list = []
evector_list = []; evalue_list = []
listje = []; listje2 = []


for i, k in enumerate(k_vals):
    eval, evec = compute_eig(k)
    evector_list.append(evec)
    evalue_list.append(eval)

    eigval1 = eval[0]
    eigval2 = eval[1]

    E1_x_plus = (evec[0, 0] * np.exp(1j * (k + n*g) * x_vals) - evec[0, 1] * np.exp(1j * (k - n*g) * x_vals))
    E2_x_plus = (evec[1, 0] * np.exp(1j * (k + n*g) * x_vals) - evec[1, 1] * np.exp(1j * (k - n*g) * x_vals))

    listje.append(h_prime_x * E1_x_plus * np.exp(1j * k * x_vals))
    listje2.append(h_prime_x * E2_x_plus * np.exp(1j * k * x_vals))

    # M1 = np.trapezoid(np.real(E1_x_plus), x_vals) # num integrate
    # M2 = np.trapezoid(np.real(E2_x_plus), x_vals)

    M1 = np.trapezoid(h_prime_x * E1_x_plus * np.exp(1j * k * x_vals), x_vals) # num integrate
    M2 = np.trapezoid(h_prime_x * E2_x_plus * np.exp(1j * k * x_vals), x_vals)

    E_idx1 = np.abs(energy_vals - eigval1).argmin()
    intensity_map[E_idx1, i] += np.abs(M1)**2 * p / L

    # M1_list.append(np.abs(M1)**2)

    E_idx2 = np.abs(energy_vals - eigval2).argmin()
    intensity_map[E_idx2, i] += np.abs(M2)**2 * p / L

    # M2_list.append(np.abs(M2)**2)

# evector_arr = np.array(evector_list)
# eval_arr = np.array(evalue_list)

# plt.plot(k_vals, evector_arr[:, 0, 1])
# plt.plot(k_vals, evector_arr[:, 1, 0])
# plt.show(); plt.close()

# plt.plot(k_vals, eval_arr[:, 0], 'o')
# plt.plot(k_vals, eval_arr[:, 1], 'o')
# plt.show(); plt.close()

intensity_map[:, [0, -1]] = 0 # remove edge intensity for convolution
intensity_map[[0, -1], :] = 0



ka = np.abs(np.trapezoid(listje, x_vals))**2
ka1 = np.abs(np.sum(listje2,axis=1))**2

plt.plot(k_vals, ka * p / L)
# plt.plot(k_vals, ka1 * p / L)
plt.show()

# for i in range(0, len(listje), 50):

#     plt.plot(listje[i])
#     plt.plot(listje2[i])
#     plt.hlines(np.sum(listje[i],axis=0), 0, len(listje[i]))
#     plt.hlines(np.sum(listje2[i],axis=0), 0, len(listje2[i]))
#     plt.show()

# plt.plot(k_vals, np.abs(ka)**2)
# plt.plot(k_vals, np.abs(ka1)**2)
# plt.show()
# # plt.plot(np.real(np.array(listje2[i])))
# plt.show()

kernel = gaussian_kernel(10, 5)[:, None] * gaussian_kernel(10, 5)
intensity_map = fftconvolve(intensity_map, kernel, mode="same")
intensity_map /= np.max(intensity_map)
plt.imshow(intensity_map, extent=[k_vals[0], k_vals[-1], energy_vals[0], energy_vals[-1]], 
           origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label="intensity")
plt.show()